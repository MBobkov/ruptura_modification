#include "DataReader.h"
#include <fstream>
#include <sstream>
#include <iostream>

bool loadStateFromFile(const std::string& filename,
                       std::vector<double>& P,
                       std::vector<double>& Q,
                       std::vector<double>& T,
                       std::vector<double>& Tw,
                       size_t Ncomp)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка: не удалось открыть файл " << filename << std::endl;
        return false;
    }

    // Считываем все строки, игнорируя комментарии и пустые
    std::vector<std::vector<double>> rawData;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        double val;
        std::vector<double> row;
        while (iss >> val) {
            row.push_back(val);
        }
        if (!row.empty()) rawData.push_back(row);
    }

    size_t nPoints = rawData.size();
    if (nPoints == 0) {
        std::cerr << "Ошибка: файл не содержит данных." << std::endl;
        return false;
    }

    // Минимальное количество колонок, необходимых для извлечения данных
    const size_t requiredCols = 17; // индексы до 16 включительно
    for (size_t i = 0; i < nPoints; ++i) {
        if (rawData[i].size() <= requiredCols) {
            std::cerr << "Ошибка в строке " << i+1 << ": недостаточно колонок." << std::endl;
            return false;
        }
    }

    // Изменяем размер выходных векторов
    P.resize(nPoints * Ncomp);
    Q.resize(nPoints * Ncomp);
    T.resize(nPoints);
    Tw.resize(nPoints);

    // Заполняем в обратном порядке: первый элемент выходных массивов (i=0)
    // соответствует последней строке файла (открытый конец).
    for (size_t i = 0; i < nPoints; ++i) {
        //size_t i_orig = nPoints - 1 - i; // индекс в исходном порядке (от z=0 до z=L)
        size_t i_orig = i;
        const auto& row = rawData[i_orig];

        // Парциальные давления и загрузки для компонентов
        // Индексы согласно описанию (считаем с 0):
        // 3: Q0, 5: P0, 9: Q1, 11: P1
        if (Ncomp >= 1) {
            Q[i * Ncomp + 0] = row[3];
            P[i * Ncomp + 0] = std::abs(row[5]);
        }
        if (Ncomp >= 2) {
            Q[i * Ncomp + 1] = row[9];
            if (row[11] < 0) {
                P[i * Ncomp + 1] = 0;
            }
            else {
                P[i * Ncomp + 1] = std::abs(row[11]);
            }

        }
        // Если компонентов больше, можно добавить аналогично, но формат не предусматривает.

        // Температуры
        T[i]  = row[15]; // Tgs
        Tw[i] = row[16]; // Tw

        //std::cout << i << " " << P[i * Ncomp + 1] << std::endl;
    }

    std::cout << "Downloaded " << nPoints << " of specified state!" << std::endl;

    return true;
}

void initializeFromFile(const std::string& filename, size_t Ncomp,
                        std::vector<double>& P,
                        std::vector<double>& Q,
                        std::vector<double>& T,
                        std::vector<double>& Tw)
{
    std::vector<double> P_loaded, Q_loaded, T_loaded, Tw_loaded;
    if (loadStateFromFile(filename, P_loaded, Q_loaded, T_loaded, Tw_loaded, Ncomp)) {
        // Перемещаем загруженные данные в предоставленные векторы
        P = std::move(P_loaded);
        Q = std::move(Q_loaded);
        T = std::move(T_loaded);
        Tw = std::move(Tw_loaded);
    } else {
        std::cerr << "Data downloading has failed!" << std::endl;
    }
}