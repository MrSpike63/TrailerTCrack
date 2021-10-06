#include <cinttypes>
#include <iostream>
#include <chrono>

#include <cuda.h>


#define THREAD_COUNT 128
#define TASK_WORK (1ULL << 30)

#define FAST_NEXT_INT


#ifdef BOINC
    #include "boinc_api.h"
    #if defined _WIN32 || defined _WIN64
        #include "boinc_win.h"
    #endif
#endif

#ifndef BOINC
    #define boinc_fopen(file, mode) fopen(file, mode)
    #define boinc_delete_file(file) remove(file)

    #define boinc_begin_critical_section()
    #define boinc_end_critical_section()

    #define boinc_fraction_done(frac)
    #define boinc_finish(s) exit(s)

    #define boinc_time_to_checkpoint() true
    #define boinc_checkpoint_completed()
#endif



namespace Random {
    #define RANDOM__MULTIPLIER 25214903917ULL
    #define RANDOM__MULTIPLIER_INVERSE 246154705703781ULL

    #define RANDOM__ADDEND 11ULL
    #define RANDOM__ADDEND_INVERSE 107048004364969ULL
    #define RANDOM__MASK ((1ULL << 48) - 1)

    __device__ uint64_t setSeed(uint64_t seed) {
        return (seed ^ RANDOM__MULTIPLIER) & RANDOM__MASK;
    }

    __device__ int32_t next(uint64_t &seed, int bits) {
        seed = (seed * RANDOM__MULTIPLIER + RANDOM__ADDEND) & RANDOM__MASK;

        return (int32_t)(seed >> (48 - bits));
    }

    __device__ int32_t nextInt(uint64_t &seed) {
        return next(seed, 32);
    }

    __device__ int32_t nextInt(uint64_t &seed, int bound) {
        if ((bound & -bound) == bound) {
            seed = (seed * RANDOM__MULTIPLIER + RANDOM__ADDEND) & RANDOM__MASK;
            return (int32_t)((bound * (seed >> 17)) >> 31);
        }
    
        int32_t bits, value;
        #ifndef FAST_NEXT_INT
        do {
        #endif
            seed = (seed * RANDOM__MULTIPLIER + RANDOM__ADDEND) & RANDOM__MASK;
            bits = seed >> 17;
            value = bits % bound;
        
        #ifndef FAST_NEXT_INT
        } while (bits - value + (bound - 1) < 0);
        #endif
        return value;
    }

    __device__ uint64_t nextLong(uint64_t &seed) {
        return ((uint64_t)next(seed, 32) << 32) + next(seed, 32);
    }

    __device__ float nextFloat(uint64_t &seed) {
        return next(seed, 24) / ((float)(1 << 24));
    }

    __device__ double nextDouble(uint64_t &seed) {
        return (((uint64_t)next(seed, 26) << 27) + next(seed, 27)) / (double)(1ULL << 53);
    }

    template <int n>
    __device__ constexpr void advance(uint64_t &seed) {
        uint64_t m = 1;
        uint64_t a = 0;
        for (int i = 0; i < n; i++) {
            a = (a * RANDOM__MULTIPLIER + RANDOM__ADDEND) & RANDOM__MASK;
            m = (m * RANDOM__MULTIPLIER) & RANDOM__MASK;
        }

        seed = (seed * m + a) & RANDOM__MASK;
    }
}


__shared__ uint8_t sharedMemory[256 * THREAD_COUNT];
#define SHARED_MEMORY_ACCESS(n) sharedMemory[(threadIdx.x << 8) | n]
#define CASTED_SHARED_MEMORY_ACCESS(n) ((double*)sharedMemory)[(threadIdx.x << 5) | n]


namespace Terrain {
    struct OctaveData {
        double xOffset;
        double yOffset;
        double zOffset;
        uint8_t permutations[256];
    };

    struct NoiseData {
        OctaveData noise1[16];
        OctaveData noise2[16];
        OctaveData noise3[8];
        OctaveData noise6[16];
    };

    __device__ void initializeOctave(uint64_t &random, OctaveData *octaveData) {
        octaveData->xOffset = Random::nextDouble(random) * 256.0;
        octaveData->yOffset = Random::nextDouble(random) * 256.0;
        octaveData->zOffset = Random::nextDouble(random) * 256.0;


        for (int i = 0; i < 256; i++) {
            SHARED_MEMORY_ACCESS(i) = i;
        }

        for (int i = 0; i < 256; i++) {
            uint8_t k = Random::nextInt(random, 256 - i) + i;
            uint8_t l = SHARED_MEMORY_ACCESS(i);
            octaveData->permutations[i] = SHARED_MEMORY_ACCESS(k);
            SHARED_MEMORY_ACCESS(k) = l;
        }
    }

    __device__ void initializeNoise(uint64_t worldSeed, NoiseData* noiseData) {
        uint64_t random = Random::setSeed(worldSeed);

        for (int i = 0; i < 16; i++) { initializeOctave(random, &noiseData->noise1[i]); }
        for (int i = 0; i < 16; i++) { initializeOctave(random, &noiseData->noise2[i]); }
        for (int i = 0; i < 8; i++) { initializeOctave(random, &noiseData->noise3[i]); }


        #ifndef FAST_NEXT_INT
            for (int i = 0; i < 14; i++) {
                Random::advance<7>(random);
                for (int j = 1; j < 256; j++) {
                    Random::nextInt(random, 256 - j);
                }
            }
        #else
            Random::advance<3668>(random);
        #endif

        for (int i = 0; i < 16; i++) { initializeOctave(random, &noiseData->noise6[i]); }
    }



    __device__ double lerp(double t, double a, double b) {
        return a + t * (b - a);
    }



    __device__ double func_4110_a(int i, double x, double z) {
        switch (i & 0xF) {
            case 0x0:
                return x;
            case 0x1:
                return -x;
            case 0x2:
                return x;
            case 0x3:
                return -x;
            case 0x4:
                return x + z;
            case 0x5:
                return -x + z;
            case 0x6:
                return x - z;
            case 0x7:
                return -x - z;
            case 0x8:
                return z;
            case 0x9:
                return -z;
            case 0xA:
                return -z;
            case 0xB:
                return -z;
            case 0xC:
                return x;
            case 0xD:
                return z;
            case 0xE:
                return -x;
            case 0xF:
                return -z;
            default:
                return 0;
        }
    }


    __device__ double grad(int i, double x, double y, double z) {
        switch (i & 0xF) {
            case 0x0:
                return x + y;
            case 0x1:
                return -x + y;
            case 0x2:
                return x - y;
            case 0x3:
                return -x - y;
            case 0x4:
                return x + z;
            case 0x5:
                return -x + z;
            case 0x6:
                return x - z;
            case 0x7:
                return -x - z;
            case 0x8:
                return y + z;
            case 0x9:
                return -y + z;
            case 0xA:
                return y - z;
            case 0xB:
                return -y - z;
            case 0xC:
                return y + x;
            case 0xD:
                return -y + z;
            case 0xE:
                return y - x;
            case 0xF:
                return -y - z;
            default:
                return 0;
        }
    }


    __device__ uint8_t getPermutation(const uint8_t* __restrict__ permutations, int n) {
        return permutations[n & 0xFF];
    }


    __device__ double optimizedNoise2D(const OctaveData* __restrict__ octaveDatas, double baseX, double baseZ, int xIteration, int zIteration, double noiseScaleX, double noiseScaleZ, int numOctaves) {
        double outputValue = 0;

        double octavesFactor = 1.0;
        for (int i = 0; i < numOctaves; i++) {
            double noiseFactorX = noiseScaleX * octavesFactor;
            double noiseFactorZ = noiseScaleZ * octavesFactor;

            double startX = (double)baseX * octavesFactor * noiseScaleX;
            double startZ = (double)baseZ * octavesFactor * noiseScaleZ;

            double octaveWidth = 1.0 / octavesFactor;

            double xCoord = startX + (double)xIteration * noiseFactorX + octaveDatas[i].xOffset;
            int xCoordFloor = (int)xCoord;
            if (xCoord < (double)xCoordFloor) {
                xCoordFloor--;
            }
            int xUnitCube = xCoordFloor & 0xFF;
            xCoord -= xCoordFloor;
            double fadeX = xCoord * xCoord * xCoord * (xCoord * (xCoord * 6.0 - 15.0) + 10.0);

            double zCoord = startZ + (double)zIteration * noiseFactorZ + octaveDatas[i].zOffset;
            int zCoordFloor = (int)zCoord;
            if (zCoord < (double)zCoordFloor) {
                zCoordFloor--;
            }
            int zUnitCube = zCoordFloor & 0xFF;
            zCoord -= zCoordFloor;
            double fadeZ = zCoord * zCoord * zCoord * (zCoord * (zCoord * 6.0 - 15.0) + 10.0);

            int l = getPermutation(octaveDatas[i].permutations, xUnitCube) + 0;
            int j1 = getPermutation(octaveDatas[i].permutations, l) + zUnitCube;
            int k1 = getPermutation(octaveDatas[i].permutations, xUnitCube + 1) + 0;
            int l1 = getPermutation(octaveDatas[i].permutations, k1) + zUnitCube;

            double d9 = lerp(fadeX, func_4110_a(getPermutation(octaveDatas[i].permutations, j1), xCoord, zCoord), grad(getPermutation(octaveDatas[i].permutations, l1), xCoord - 1.0, 0.0, zCoord));
            double d11 = lerp(fadeX, grad(getPermutation(octaveDatas[i].permutations, j1 + 1), xCoord, 0.0, zCoord - 1.0), grad(getPermutation(octaveDatas[i].permutations, l1 + 1), xCoord - 1.0, 0.0, zCoord - 1.0));
            double d23 = lerp(fadeZ, d9, d11);
            outputValue += d23 * octaveWidth;


            octavesFactor /= 2.0;
        }

        return outputValue;
    }

    __device__ void optimizedNoise3D(const OctaveData* __restrict__ octaveDatas, int sharedMemoryOffset, double baseX, double baseY, double baseZ, int xIteration, int zIteration, double noiseScaleX, double noiseScaleY, double noiseScaleZ, int numOctaves, int yIterationStart, int yIterations) {
        double octavesFactor = 1.0;
        for (int i = 0; i < numOctaves; i++) {
            double noiseFactorX = noiseScaleX * octavesFactor;
            double noiseFactorY = noiseScaleY * octavesFactor;
            double noiseFactorZ = noiseScaleZ * octavesFactor;

            double startX = (double)baseX * octavesFactor * noiseScaleX;
            double startY = (double)baseY * octavesFactor * noiseScaleY;
            double startZ = (double)baseZ * octavesFactor * noiseScaleZ;

            int i2 = -1;
            double d13 = 0.0;
            double d15 = 0.0;
            double d16 = 0.0;
            double d18 = 0.0;

            double octaveWidth = 1.0 / octavesFactor;

            double xCoord = startX + (double)xIteration * noiseFactorX + octaveDatas[i].xOffset;
            int xCoordFloor = (int)xCoord;
            if (xCoord < (double)xCoordFloor) {
                xCoordFloor--;
            }
            int xUnitCube = xCoordFloor & 0xFF;
            xCoord -= xCoordFloor;
            double fadeX = xCoord * xCoord * xCoord * (xCoord * (xCoord * 6.0 - 15.0) + 10.0);

            double zCoord = startZ + (double)zIteration * noiseFactorZ + octaveDatas[i].zOffset;
            int zCoordFloor = (int)zCoord;
            if (zCoord < (double)zCoordFloor) {
                zCoordFloor--;
            }
            int zUnitCube = zCoordFloor & 0xFF;
            zCoord -= zCoordFloor;
            double fadeZ = zCoord * zCoord * zCoord * (zCoord * (zCoord * 6.0 - 15.0) + 10.0);

            for (int yIteration = 0; yIteration < yIterationStart + yIterations; yIteration++) {
                double yCoord = startY + (double)yIteration * noiseFactorY + octaveDatas[i].yOffset;
                int yCoordFloor = (int)yCoord;
                if (yCoord < (double)yCoordFloor) {
                    yCoordFloor--;
                }
                int yUnitCube = yCoordFloor & 0xFF;
                yCoord -= yCoordFloor;
                double fadeY = yCoord * yCoord * yCoord * (yCoord * (yCoord * 6.0 - 15.0) + 10.0);

                if (yIteration == 0 || yUnitCube != i2) {
                    i2 = yUnitCube;
                    int j2 = getPermutation(octaveDatas[i].permutations, xUnitCube) + yUnitCube;
                    int k2 = getPermutation(octaveDatas[i].permutations, j2) + zUnitCube;
                    int l2 = getPermutation(octaveDatas[i].permutations, j2 + 1) + zUnitCube;
                    int i3 = getPermutation(octaveDatas[i].permutations, xUnitCube + 1) + yUnitCube;
                    int k3 = getPermutation(octaveDatas[i].permutations, i3) + zUnitCube;
                    int l3 = getPermutation(octaveDatas[i].permutations, i3 + 1) + zUnitCube;
                    d13 = lerp(fadeX, grad(getPermutation(octaveDatas[i].permutations, k2), xCoord, yCoord, zCoord), grad(getPermutation(octaveDatas[i].permutations, k3), xCoord - 1.0, yCoord, zCoord));
                    d15 = lerp(fadeX, grad(getPermutation(octaveDatas[i].permutations, l2), xCoord, yCoord - 1.0, zCoord), grad(getPermutation(octaveDatas[i].permutations, l3), xCoord - 1.0, yCoord - 1.0, zCoord));
                    d16 = lerp(fadeX, grad(getPermutation(octaveDatas[i].permutations, k2 + 1), xCoord, yCoord, zCoord - 1.0), grad(getPermutation(octaveDatas[i].permutations, k3 + 1), xCoord - 1.0, yCoord, zCoord - 1.0));
                    d18 = lerp(fadeX, grad(getPermutation(octaveDatas[i].permutations, l2 + 1), xCoord, yCoord - 1.0, zCoord - 1.0), grad(getPermutation(octaveDatas[i].permutations, l3 + 1), xCoord - 1.0, yCoord - 1.0, zCoord - 1.0));
                }
                double d28 = lerp(fadeY, d13, d15);
                double d29 = lerp(fadeY, d16, d18);
                double d30 = lerp(fadeZ, d28, d29);

                if (yIteration >= yIterationStart) {
                    CASTED_SHARED_MEMORY_ACCESS(yIteration - yIterationStart + sharedMemoryOffset) += d30 * octaveWidth;
                }
            }


            octavesFactor /= 2.0;
        }
    }


    __device__ void mixNoiseValues(int sharedMemoryOutputOffset, int sharedMemoryNoise1Offset, int sharedMemoryNoise2Offset, int sharedMemoryNoise3Offset, double noise6, int yAreaStart, int yAreas) {
        int i2 = 0;
        int j2 = 0;

        float f1 = 0.37000000476837158203125f;
        float f2 = -0.07500000298023223876953125;

        double d2 = noise6 / 8000.0;
        if (d2 < 0.0) {
            d2 = -d2 * 0.29999999999999999;
        }
        d2 = d2 * 3.0 - 2.0;
        if (d2 < 0.0) {
            d2 /= 2.0;
            if (d2 < -1.0) {
                d2 = -1.0;
            }
            d2 /= 1.3999999999999999;
            d2 /= 2.0;
        } else {
            if (d2 > 1.0) {
                d2 = 1.0;
            }
            d2 /= 8.0;
        }
        j2++;
        for (int k3 = yAreaStart; k3 < (yAreaStart + yAreas); k3++) {
            double d3 = f2;
            double d4 = f1;
            d3 += d2 * 0.20000000000000001;
            d3 = (d3 * (double)17) / 16.0;
            double d5 = (double)17 / 2.0 + d3 * 4.0;
            double d6 = 0.0;
            double d7 = (((double)k3 - d5) * 12.0 * 128.0) / (double)(1 << 7) / d4;
            if (d7 < 0.0) {
                d7 *= 4.0;
            }
            double d8 = CASTED_SHARED_MEMORY_ACCESS(i2 + sharedMemoryNoise1Offset) / 512.0;
            double d9 = CASTED_SHARED_MEMORY_ACCESS(i2 + sharedMemoryNoise2Offset) / 512.0;
            double d10 = (CASTED_SHARED_MEMORY_ACCESS(i2 + sharedMemoryNoise3Offset) / 10.0 + 1.0) / 2.0;

            if (d10 < 0.0) {
                d6 = d8;
            } else if (d10 > 1.0) {
                d6 = d9;
            } else {
                d6 = d8 + (d9 - d8) * d10;
            }

            d6 -= d7;
            if (k3 > 17 - 4) {
                double d11 = (float)(k3 - (17 - 4)) / 3.0f;
                d6 = d6 * (1.0 - d11) + -10.0 * d11;
            }
            CASTED_SHARED_MEMORY_ACCESS(i2 + sharedMemoryOutputOffset) = d6;
            i2++;
        }
    }

    __device__ void optimizedNoise(const NoiseData* __restrict__ noiseData, int sharedMemoryWriteOffset, int32_t x, int32_t y, int32_t z, int xArea, int zArea, int yAreaStart, int yAreas) {
        double noise6Value = optimizedNoise2D(noiseData->noise6, (double)x, (double)z, xArea, zArea, 200.0, 200.0, 16);

        for (int i = 0; i < yAreas; i++) {
            CASTED_SHARED_MEMORY_ACCESS(i) = 0.0;
        }
        for (int i = 0; i < yAreas; i++) {
            CASTED_SHARED_MEMORY_ACCESS(i + yAreas) = 0.0;
        }
        for (int i = 0; i < yAreas; i++) {
            CASTED_SHARED_MEMORY_ACCESS(i + yAreas + yAreas) = 0.0;
        }
        

        optimizedNoise3D(noiseData->noise1, 0, (double)x, (double)y, (double)z, xArea, zArea, 684.41200000000003, 684.41200000000003, 684.41200000000003, 16, yAreaStart, yAreas);
        optimizedNoise3D(noiseData->noise2, yAreas, (double)x, (double)y, (double)z, xArea, zArea, 684.41200000000003, 684.41200000000003, 684.41200000000003, 16, yAreaStart, yAreas);
        optimizedNoise3D(noiseData->noise3, yAreas + yAreas, (double)x, (double)y, (double)z, xArea, zArea, 8.5551500000000011, 4.2775750000000006, 8.5551500000000011, 8, yAreaStart, yAreas);
        mixNoiseValues(sharedMemoryWriteOffset, 0, yAreas, yAreas + yAreas, noise6Value, yAreaStart, yAreas);
    }

    __device__ void optimizedPointLerp(int sharedMemoryOffset, double bottomRight, double bottomLeft, double topRight, double topLeft, double bottomRight2, double bottomLeft2, double topRight2, double topLeft2, uint8_t baseHeight) {
        double bottomRightDiff = (bottomRight2 - bottomRight) * 0.125;
        double bottomLeftDiff = (bottomLeft2 - bottomLeft) * 0.125;
        double topRightDiff = (topRight2 - topRight) * 0.125;
        double topLeftDiff = (topLeft2 - topLeft) * 0.125;

        for (int y = 0; y < 8; y++) {
            double localBottomRight = bottomRight;
            double localTopRight = topRight;

            double localBottomRightDiff = (bottomLeft - bottomRight) * 0.25;
            double localTopRightDiff = (topLeft - topRight) * 0.25;

            for (int x = 0; x < 4; x++) {
                double localHeight = localBottomRight;
                double zStep = (localTopRight - localBottomRight) * 0.25;

                localHeight -= zStep;

                for (int z = 0; z < 4; z++) {
                    if ((localHeight += zStep) > 0.0) {
                        SHARED_MEMORY_ACCESS(x * 4 + z + sharedMemoryOffset) = baseHeight + y;
                    }
                }

                localBottomRight += localBottomRightDiff;
                localTopRight += localTopRightDiff;
            }

            bottomRight += bottomRightDiff;
            bottomLeft += bottomLeftDiff;
            topRight += topRightDiff;
            topLeft += topLeftDiff;
        }
    }

    __device__ uint8_t optimizedMod4Lerp(double a, double b, uint8_t baseHeight) {
        uint8_t height = 0;
        double diff = (b - a) * 0.125;
        for (int i = 0; i < 8; i++) {
            if (a > 0) {
                height = baseHeight + i;
            }
            a += diff;
        }

        return height;
    }
}


__device__ bool checkTerrain(uint64_t worldSeed) {
    Terrain::NoiseData noiseData;
    Terrain::initializeNoise(worldSeed, &noiseData);

    Terrain::optimizedNoise(&noiseData, 9, -22 * 4, 0, 2 * 4, 0, 2, 8, 2);
    if (Terrain::optimizedMod4Lerp(CASTED_SHARED_MEMORY_ACCESS(9), CASTED_SHARED_MEMORY_ACCESS(10), 64) != 65) { 
        return false;
    }

    Terrain::optimizedNoise(&noiseData, 11, -22 * 4, 0, 2 * 4, 1, 2, 8, 2);
    if (Terrain::optimizedMod4Lerp(CASTED_SHARED_MEMORY_ACCESS(11), CASTED_SHARED_MEMORY_ACCESS(12), 64) != 67) { 
        return false;
    }

    Terrain::optimizedNoise(&noiseData, 13, -22 * 4, 0, 2 * 4, 0, 3, 8, 2);
    if (Terrain::optimizedMod4Lerp(CASTED_SHARED_MEMORY_ACCESS(13), CASTED_SHARED_MEMORY_ACCESS(14), 64) != 67) { 
        return false;
    }

    Terrain::optimizedNoise(&noiseData, 15, -22 * 4, 0, 2 * 4, 1, 3, 7, 3);
    if (CASTED_SHARED_MEMORY_ACCESS(16) > 0) { return false; }
    if (Terrain::optimizedMod4Lerp(CASTED_SHARED_MEMORY_ACCESS(15), CASTED_SHARED_MEMORY_ACCESS(16), 56) != 63) { 
        return false;
    }
    
    Terrain::optimizedNoise(&noiseData, 18, -22 * 4, 0, 2 * 4, 2, 3, 7, 2);
    if (CASTED_SHARED_MEMORY_ACCESS(19) > 0) { return false; }
    if (Terrain::optimizedMod4Lerp(CASTED_SHARED_MEMORY_ACCESS(18), CASTED_SHARED_MEMORY_ACCESS(19), 56) != 63) { 
        return false;
    }


    
    int sharedMemoryOffset = 0;
    for (int i = 0; i < 16; i++) {
        SHARED_MEMORY_ACCESS(sharedMemoryOffset + i) = 0; 
    }
    Terrain::optimizedPointLerp(sharedMemoryOffset, CASTED_SHARED_MEMORY_ACCESS(9), CASTED_SHARED_MEMORY_ACCESS(11), CASTED_SHARED_MEMORY_ACCESS(13), CASTED_SHARED_MEMORY_ACCESS(16), CASTED_SHARED_MEMORY_ACCESS(10), CASTED_SHARED_MEMORY_ACCESS(12), CASTED_SHARED_MEMORY_ACCESS(14), CASTED_SHARED_MEMORY_ACCESS(17), 64);

    if (SHARED_MEMORY_ACCESS(sharedMemoryOffset + 2) != 66) { return false; }
    if (SHARED_MEMORY_ACCESS(sharedMemoryOffset + 3) != 67) { return false; }
    if (SHARED_MEMORY_ACCESS(sharedMemoryOffset + 4) != 65) { return false; }
    if (SHARED_MEMORY_ACCESS(sharedMemoryOffset + 6) != 66) { return false; }
    if (SHARED_MEMORY_ACCESS(sharedMemoryOffset + 7) != 66) { return false; }
    if (SHARED_MEMORY_ACCESS(sharedMemoryOffset + 8) != 65) { return false; }
    // if (SHARED_MEMORY_ACCESS(sharedMemoryOffset + 9) != 65) { return false; }
    if (SHARED_MEMORY_ACCESS(sharedMemoryOffset + 12) != 66) { return false; }
    if (SHARED_MEMORY_ACCESS(sharedMemoryOffset + 13) != 65) { return false; }
    if (SHARED_MEMORY_ACCESS(sharedMemoryOffset + 14) != 64) { return false; }
    if (SHARED_MEMORY_ACCESS(sharedMemoryOffset + 15) != 64) { return false; }

    return true;
}


__device__ __managed__ uint32_t outputCounter = 0;
__device__ __managed__ uint64_t outputBuffer[100000];
__global__ void __launch_bounds__(THREAD_COUNT, 3) gpuWork(uint64_t seedOffset) {
    uint64_t worldSeed = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x + seedOffset;

    if (!checkTerrain(worldSeed)) {
        return;
    }

    uint32_t idx = atomicAdd(&outputCounter, 1);
    outputBuffer[idx] = worldSeed;
}


uint64_t milliseconds() {
    return (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())).count();
}

#define GPU_ASSERT(code) gpuAssert((code), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s (code %d) %s %d\n", cudaGetErrorString(code), code, file, line);
        boinc_finish(code);
    }
}


int calculateBlockSize(double threshold) {
    gpuWork<<<1, THREAD_COUNT>>>(0);
    GPU_ASSERT(cudaPeekAtLastError());
    GPU_ASSERT(cudaDeviceSynchronize());
    GPU_ASSERT(cudaPeekAtLastError());
    outputCounter = 0;

    int setBits = 0;
    int lowestSetBit = 30;
    for (int i = 0; i < 30; i++) {
        int j;
        for (j = 0; j < lowestSetBit; j++) {
            int32_t newBits = setBits | (1 << j);

            uint64_t startTime = milliseconds();

            gpuWork<<<newBits, THREAD_COUNT>>>(0);
            GPU_ASSERT(cudaPeekAtLastError());
            GPU_ASSERT(cudaDeviceSynchronize());
            GPU_ASSERT(cudaPeekAtLastError());
            outputCounter = 0;

            uint64_t endTime = milliseconds();

            double elapsed = (double)(endTime - startTime) / 1000.0;
            
            if (elapsed > threshold) {
                if (j != 0) {
                    setBits |= (1 << (j - 1));
                    lowestSetBit = (j - 1);
                } else if (j == 0) {
                    lowestSetBit = 0;
                }
                break;
            }
        }

        if (lowestSetBit == 0) { break; }

        if (j == lowestSetBit) {
            setBits |= (1 << (j - 1));
            lowestSetBit = (j - 1);
        }
    }

    return setBits;
}


struct CheckpointData {
    int lastIteration;
    double elapsed;
    int blockCount;
};

int main(int argc, char* argv[]) {
    int taskNumber = 0;
    int device = 0;
    for (int i = 1; i < argc; i += 2) {
        const char *param = argv[i];
        if (strcmp(param, "-t") == 0 || strcmp(param, "--task") == 0) {
            taskNumber = atoi(argv[i + 1]);
        } else if (strcmp(param, "-d") == 0 || strcmp(param, "--device") == 0) {
            device = atoi(argv[i + 1]);
        }
    }

    int startIteration = 0;
    double elapsed = 0;
    int BLOCK_COUNT = 0;

    fprintf(stderr, "Recieved work unit: %d.\n", taskNumber);
    fflush(stderr);

    #ifdef BOINC
        BOINC_OPTIONS options;
        boinc_options_defaults(options);
        options.normal_thread_priority = true; 
        boinc_init_options(&options);

        APP_INIT_DATA aid;
        boinc_get_init_data(aid);
        
        if (aid.gpu_device_num >= 0) {
            fprintf(stderr, "boinc gpu: %d, cli gpu: %d.\n", aid.gpu_device_num, device);
            device = aid.gpu_device_num;
        } else {
            fprintf(stderr, "cli gpu: %d.\n", device);
        }
    #endif

    cudaSetDevice(device);
    GPU_ASSERT(cudaPeekAtLastError());
    GPU_ASSERT(cudaDeviceSynchronize());
    GPU_ASSERT(cudaPeekAtLastError());

    FILE* checkpointFile = boinc_fopen("trailer_checkpoint.txt", "rb");

    if (checkpointFile) {
        boinc_begin_critical_section();
        struct CheckpointData checkpointData;

        fread(&checkpointData, sizeof(checkpointData), 1, checkpointFile);
        startIteration = checkpointData.lastIteration + 1;
        elapsed = checkpointData.elapsed;
        BLOCK_COUNT = checkpointData.blockCount;

        fclose(checkpointFile);
        fprintf(stderr, "Loaded checkpoint %d %.2f %d.\n", startIteration, elapsed, BLOCK_COUNT);
        fflush(stderr);
        boinc_end_critical_section();
    } else {
        fprintf(stderr, "No checkpoint to load.\n");
    }
    if (BLOCK_COUNT == 0) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        int cc = deviceProp.major * 10 + deviceProp.minor;

        if (cc <= 52) {
            BLOCK_COUNT = calculateBlockSize(0.02);
        } else if (deviceProp.major == 6) {
            BLOCK_COUNT = calculateBlockSize(0.1);
        } else if (deviceProp.major == 7) {
            BLOCK_COUNT = calculateBlockSize(0.15);
        } else if (deviceProp.major == 8) {
            BLOCK_COUNT = calculateBlockSize(0.5);
        } else {
            fprintf(stderr, "Unrecognized compute capability.\n");
            fflush(stderr);
            boinc_finish(1);
        }
        fprintf(stderr, "Calculated block count: %d.\n", BLOCK_COUNT);
        if (BLOCK_COUNT == 0) { BLOCK_COUNT = 1; }
        fflush(stderr);
    }

    uint64_t GRID_WORK = (uint64_t)BLOCK_COUNT * THREAD_COUNT;
    int ITERATIONS_NEEDED = ((TASK_WORK + GRID_WORK - 1) / GRID_WORK);

    for (int i = startIteration; i < ITERATIONS_NEEDED; i++) {
        uint64_t seedOffset = (TASK_WORK * taskNumber) + GRID_WORK * i;
        uint64_t startTime = milliseconds();

        gpuWork<<<BLOCK_COUNT, THREAD_COUNT>>>(seedOffset);
        GPU_ASSERT(cudaPeekAtLastError());
        GPU_ASSERT(cudaDeviceSynchronize());
        GPU_ASSERT(cudaPeekAtLastError());

        uint64_t endTime = milliseconds();

        boinc_begin_critical_section();

        double localElapsed = ((double)(endTime - startTime) / 1000);
        elapsed += localElapsed;

        if (boinc_time_to_checkpoint()) {
            struct CheckpointData checkpointData;
            checkpointData.lastIteration = i;
            checkpointData.elapsed = elapsed;
            checkpointData.blockCount = BLOCK_COUNT;

            FILE* checkpointFile = boinc_fopen("trailer_checkpoint.txt", "wb");
            fwrite(&checkpointData, sizeof(checkpointData), 1, checkpointFile);
            fclose(checkpointFile);

            boinc_checkpoint_completed();
        }

        if (outputCounter > 0) {
            FILE *seedsOut = boinc_fopen("trailer_seeds.txt", "a");
            for (int j = 0; j < outputCounter; j++) {
                if (outputBuffer[j] < (TASK_WORK * (taskNumber + 1))) {
                    fprintf(seedsOut, "Seed: %llu\n", outputBuffer[j]);
                }
            }
            fclose(seedsOut);
            outputCounter = 0;
        }

        double fracDone = (double)i / ITERATIONS_NEEDED;
        boinc_fraction_done(fracDone);

        boinc_end_critical_section();   
    }

    boinc_begin_critical_section();
    FILE *seedsOut = boinc_fopen("trailer_seeds.txt", "a");
    fclose(seedsOut);

    fprintf(stderr, "Finished in %.2f seconds. Speed: %.2f/s.\n", elapsed, (double)TASK_WORK / elapsed);
    fflush(stderr);
    boinc_delete_file("trailer_checkpoint.txt");

    boinc_end_critical_section();

    boinc_finish(0);
}