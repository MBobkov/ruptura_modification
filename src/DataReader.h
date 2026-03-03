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

#endif // DATAREADER_H