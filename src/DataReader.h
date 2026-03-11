#ifndef DATAREADER_H
#define DATAREADER_H

#include <string>
#include <vector>

bool loadStateFromFile(const std::string& filename,
                       std::vector<double>& P,
                       std::vector<double>& Q,
                       std::vector<double>& T,
                       std::vector<double>& Tw,
                       size_t Ncomp);  // изменено на size_t

void initializeFromFile(const std::string& filename, size_t Ncomp,
                        std::vector<double>& P,
                        std::vector<double>& Q,
                        std::vector<double>& T,
                        std::vector<double>& Tw);


bool loadStateFromFilePurge(const std::string& filename,
                       std::vector<double>& P,
                       std::vector<double>& Q,
                       std::vector<double>& T,
                       std::vector<double>& Tw,
                       size_t Ncomp);

void initializeFromFilePurge(const std::string& filename, size_t Ncomp,
                        std::vector<double>& P,
                        std::vector<double>& Q,
                        std::vector<double>& T,
                        std::vector<double>& Tw);

void reverseGridData(std::vector<double>& data, size_t ncomp, size_t ngrid);

#endif // DATAREADER_H