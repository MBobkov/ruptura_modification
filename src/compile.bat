@echo off
cl /std:c++17 /W4 /EHsc /O2 /MD /Zc:__cplusplus /fp:fast /arch:AVX2 *.cpp /link /out:ruptura.exe
