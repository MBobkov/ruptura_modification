#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#if __cplusplus >= 201703L && __has_include(<filesystem>)
#include <filesystem>
#elif __cplusplus >= 201703L && __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
#else
#include <sys/stat.h>
#endif

#include "pressurization.h"

#include "mixture_prediction.h"

#ifdef PYBUILD
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif  // PYBUILD

const double R = 8.31446261815324;

const double TwallInit = 295;

const double pi = 3.1415926535;

//const double Tamb = 295;

inline double maxVectorDifference(const std::vector<double> &v, const std::vector<double> &w)
{
  if (v.empty() || w.empty()) return 0.0;
  if (v.size() != w.size()) throw std::runtime_error("Error: unequal vector size\n");

  double max = std::abs(v[0] - w[0]);
  for (size_t i = 1; i < v.size(); ++i)
  {
    double temp = std::abs(v[i] - w[i]);
    if (temp > max) max = temp;
  }
  return max;
}

// allow std::pairs to be added
template <typename T, typename U>
std::pair<T, U> operator+(const std::pair<T, U> &l, const std::pair<T, U> &r)
{
  return {l.first + r.first, l.second + r.second};
}
template <typename T, typename U>
std::pair<T, U> &operator+=(std::pair<T, U> &l, const std::pair<T, U> &r)
{
  l.first += r.first;
  l.second += r.second;
  return l;
}

Pressurization::Pressurization(const InputReader &inputReader)
    : displayName(inputReader.displayName),
      components(inputReader.components),
      carrierGasComponent(inputReader.carrierGasComponent),
      Ncomp(components.size()),
      Ngrid(inputReader.numberOfGridPoints),
      printEvery(inputReader.printEvery),
      writeEvery(inputReader.writeEvery),
      T(inputReader.temperature),
      Tamb(inputReader.Tamb),
      p_total(inputReader.totalPressure),
      dptdx(inputReader.pressureGradient),
      epsilon(inputReader.columnVoidFraction),
      epsilon1(inputReader.columnVoidFraction_1),
      rho_p(inputReader.particleDensity),
      rho_p1(inputReader.particleDensity1),
      rho_wall(inputReader.rho_w),
      lambda_ax(inputReader.lambda_ax),
      lambda_ax_1(inputReader.lambda_ax_1),
      lambda_w(inputReader.lambda_w),
      D_column_out(inputReader.D_column_out),
      D_column_inner(inputReader.D_column_inner),
      h_in(inputReader.h_in),
      h_in_1(inputReader.h_in_1),
      h_out(inputReader.h_out),
      Cps(inputReader.Cps),
      Cps_1(inputReader.Cps_1),
      Cpw(inputReader.Cpw),
      boundaryCoordinate(inputReader.boundary_coord),
      ramp_time(inputReader.RampTimePr),
      Pmin(inputReader.Pmin),
      v_in(inputReader.columnEntranceVelocity),
      L(inputReader.columnLength),
      dx(L / static_cast<double>(Ngrid)),
      dt(inputReader.timeStepPr),
      Nsteps(inputReader.numberOfTimeSteps),
      autoSteps(inputReader.autoNumberOfTimeSteps),
      pulse(inputReader.pulseBreakthrough),
      tpulse(inputReader.pulseTime),
      mixture(inputReader),
      mixture1(inputReader),
      maxIsothermTerms(inputReader.maxIsothermTerms),
      maxIsothermTerms1(inputReader.maxIsothermTerms1),
      prefactorLeft(Ncomp),
      prefactorLeftGP(Ncomp),
      prefactorRightGP(Ncomp),
      prefactorRight(Ncomp),
      Yi(Ncomp),
      Yi1(Ncomp),
      Xi(Ncomp),
      Xi1(Ncomp),
      Ni(Ncomp),
      Ni1(Ncomp),
      V(Ngrid + 1),
      Vnew(Ngrid + 1),
      Pt(Ngrid + 1),
      P((Ngrid + 1) * Ncomp),
      Pnew((Ngrid + 1) * Ncomp),
      Q((Ngrid + 1) * Ncomp),
      Qnew((Ngrid + 1) * Ncomp),
      Qeq((Ngrid + 1) * Ncomp),
      Qeq1((Ngrid + 1) * Ncomp),
      Qeqnew((Ngrid + 1) * Ncomp),
      Qeqnew1((Ngrid + 1) * Ncomp),
      Dpdt((Ngrid + 1) * Ncomp),
      Dpdtnew((Ngrid + 1) * Ncomp),
      Dqdt((Ngrid + 1) * Ncomp),
      Dqdtnew((Ngrid + 1) * Ncomp),
      DTdt(Ngrid + 1),
      DTdtnew(Ngrid + 1),
      DTdtWall(Ngrid + 1),
      DTdtWallnew(Ngrid + 1),
      cachedP0((Ngrid + 1) * Ncomp * maxIsothermTerms),
      cachedP01((Ngrid + 1) * Ncomp * maxIsothermTerms),
      cachedPsi((Ngrid + 1) * maxIsothermTerms),
      cachedPsi1((Ngrid + 1) * maxIsothermTerms),
      Tgs(Ngrid + 1),
      Tgsnew(Ngrid + 1),
      Tw(Ngrid + 1),
      Twnew(Ngrid + 1),
      Mol_mix(Ngrid + 1),
      Cpg_mix(Ngrid + 1),
      DPtdt(Ngrid + 1),
      CarrierGasExistance(inputReader.CarrierGasExistance),
      IsothermalRegime(inputReader.IsothermalRegime),
      cycle(inputReader.Cycle)
{

  //std::cout << "IN CONSTRUCTOR!" << std::endl;
  indexLeft = 0;
  indexMid = 0;
  indexRight = 0;

   for (size_t i = 0; i < Ngrid + 1; ++i)
  {
    if (boundaryCoordinate -  static_cast<double>(i) * dx < 0)  {
       indexLeft = i - 1;
       indexMid = 0;
       indexRight = i;
       break;
    }

    if ((boundaryCoordinate - static_cast<double>(i) * dx == 0) && (boundaryCoordinate != L))  {
       indexLeft = i - 1;
       indexMid = i;
       indexRight = i + 1;
       std::cout << "BOUNDARY IN THE GRID POINT!!!" << std::endl;
       break;
    }

  }

  if (boundaryCoordinate == L) {
    std::cout << "NO BOUNDARY!!" << std::endl;
    indexLeft = 0;
    indexMid = 0;
    indexRight = 0;
  }

  if ( indexLeft == 0 && indexRight == 0) {
    relLeft = 1;
    relRight = 0;
  }

  else {
    dxLeft = boundaryCoordinate - static_cast<double>(indexLeft) * dx;
    dxRight = static_cast<double>(indexRight) * dx - boundaryCoordinate;
    relLeft = dxLeft / dx;
    relRight = dxRight / dx;
  }

  std::cout << "index left: " << indexLeft << std::endl;
  std::cout << "index right: " << indexRight << std::endl;


  //std::cout << indexLeft << " " << indexRight << " " << indexMid << std::endl;

}

Pressurization::Pressurization(std::string _displayName, std::vector<Component> _components, size_t _carrierGasComponent,
                            size_t _numberOfGridPoints, size_t _printEvery, size_t _writeEvery, double _temperature,
                           double _p_total, double _columnVoidFraction, double _pressureGradient,
                           double _particleDensity, double _particleDensity1, double _boundary_len, double _columnEntranceVelocity, double _columnLength, //ADDED
                           double _timeStep, size_t _numberOfTimeSteps, bool _autoSteps, bool _pulse, double _pulseTime,
                           const MixturePrediction _mixture)
    : displayName(_displayName),
      components(_components),
      carrierGasComponent(_carrierGasComponent),
      Ncomp(_components.size()),
      Ngrid(_numberOfGridPoints),
      printEvery(_printEvery),
      writeEvery(_writeEvery),
      T(_temperature),
      p_total(_p_total),
      dptdx(_pressureGradient),
      epsilon(_columnVoidFraction),
      rho_p(_particleDensity),  // ADDED
      rho_p1(_particleDensity1),
      boundaryCoordinate(_boundary_len),
      v_in(_columnEntranceVelocity),
      L(_columnLength),
      dx(L / static_cast<double>(Ngrid)),
      dt(_timeStep),
      Nsteps(_numberOfTimeSteps),
      autoSteps(_autoSteps),
      pulse(_pulse),
      tpulse(_pulseTime),
      mixture(_mixture),
      mixture1(mixture),
      maxIsothermTerms(mixture.getMaxIsothermTerms()),
      prefactorLeft(Ncomp),
      prefactorLeftGP(Ncomp),
      prefactorRightGP(Ncomp),
      prefactorRight(Ncomp),
      Yi(Ncomp),
      Yi1(Ncomp),
      Xi(Ncomp),
      Xi1(Ncomp),
      Ni(Ncomp),
      Ni1(Ncomp),
      V(Ngrid + 1),
      Vnew(Ngrid + 1),
      Pt(Ngrid + 1),
      P((Ngrid + 1) * Ncomp),
      Pnew((Ngrid + 1) * Ncomp),
      Q((Ngrid + 1) * Ncomp),
      Qnew((Ngrid + 1) * Ncomp),
      Qeq((Ngrid + 1) * Ncomp),
      Qeq1((Ngrid + 1) * Ncomp),
      Qeqnew((Ngrid + 1) * Ncomp),
      Qeqnew1((Ngrid + 1) * Ncomp),
      Dpdt((Ngrid + 1) * Ncomp),
      Dpdtnew((Ngrid + 1) * Ncomp),
      Dqdt((Ngrid + 1) * Ncomp),
      Dqdtnew((Ngrid + 1) * Ncomp),
      DTdt(Ngrid + 1),
      DTdtnew(Ngrid + 1),
      cachedP0((Ngrid + 1) * Ncomp * maxIsothermTerms),
      cachedP01((Ngrid + 1) * Ncomp * maxIsothermTerms1),
      cachedPsi((Ngrid + 1) * maxIsothermTerms),
      cachedPsi1((Ngrid + 1) * maxIsothermTerms1)
{

  //std::cout << "IN CONSTURCOTR!" << std::endl;
  indexLeft = 0;
  indexMid = 0;
  indexRight = 0;

  if (boundaryCoordinate == L / 2) {
    boundaryCoordinate = boundaryCoordinate - 0.001;
  }

  for (size_t i = 0; i < Ngrid + 1; ++i)
  {
    if (boundaryCoordinate - static_cast<double>(i) * dx < 0)  {
       indexLeft = i - 1;
       indexMid = 0;
       indexRight = i;
       break;
    }

    if ((boundaryCoordinate - static_cast<double>(i) * dx == 0) && boundaryCoordinate != L)  {
       indexLeft = i - 1;
       indexMid = i;
       indexRight = i + 1;
       break;
    }

    
  }

  if (boundaryCoordinate == L) {
    std::cout << "NO BOUNDARY!!" << std::endl;
    indexLeft = 0;
    indexMid = 0;
    indexRight = 0;
  }

  std::cout << boundaryCoordinate << std::endl;
  std::cout << L << std::endl;

  if ( indexLeft == 0 ) {
    relLeft = 1;
    relRight = 0;
  }

  else {
    dxLeft = boundaryCoordinate - static_cast<double>(indexLeft) * dx;
    dxRight = static_cast<double>(indexRight) * dx - boundaryCoordinate;
    relLeft = dxLeft / dx;
    relRight = dxRight / dx;
  }

  initialize();
}

void Pressurization::initialize()
{
  // 1. Префакторы (Mass Transfer Coefficients)
  for (size_t j = 0; j < Ncomp; ++j) {
    prefactorLeft[j] = R * ((1.0 - epsilon) / epsilon) * rho_p * components[j].Kl;
    prefactorLeftGP[j] = R * ((1.0 - epsilon) / epsilon) * ( rho_p) * ( components[j].Kl);
    prefactorRightGP[j] = R * ((1.0 - epsilon1) / epsilon1) * ( rho_p1 ) * ( components[j].Kl1 );
    prefactorRight[j] = R * ((1.0 - epsilon1) / epsilon1) * rho_p1 * components[j].Kl1; 
  }

  // 2. Очистка массивов
  std::fill(P.begin(), P.end(), 0.0);
  std::fill(Pnew.begin(), Pnew.end(), 0.0);
  std::fill(Q.begin(), Q.end(), 0.0);
  std::fill(Qnew.begin(), Qnew.end(), 0.0);

  // Температуры = 295 К
  std::fill(Tgs.begin(), Tgs.end(), T);
  std::fill(Tgsnew.begin(), Tgsnew.end(), T); 
  std::fill(Tw.begin(), Tw.end(), Tamb);
  std::fill(Twnew.begin(), Twnew.end(), Tamb);

  // Скорость = 0 (Покой)
  std::fill(V.begin(), V.end(), 0.0);
  std::fill(Vnew.begin(), Vnew.end(), 0.0);

  // 3. Заполнение колонны инертным газом (Carrier Gas)
  // Вся колонна, включая вход, заполнена инертным газом при 1 атм.
  // Примесей нет.

  if (InitialStage) {       // Need for cycle
    if (CarrierGasExistance) {
      for (size_t i = 0; i < Ngrid + 1; ++i) {
        P[i * Ncomp + carrierGasComponent] = Pmin; // 100000 Pa
        Pnew[i * Ncomp + carrierGasComponent] = Pmin;
      }
  } else {
    for (size_t i = 0; i < Ngrid + 1; ++i) {
        P[i * Ncomp + 0] = Pmin;       
        Pnew[i * Ncomp + 0] = Pmin;
    }
  }
  }
  
  

  // 4. Обновляем общее давление Pt
  for (size_t i = 0; i < Ngrid + 1; ++i) {
    Pt[i] = 0.0;
    for (size_t j = 0; j < Ncomp; ++j) Pt[i] += P[i * Ncomp + j];
  }

  // 5. Свойства смеси (Cpg) - ОБЯЗАТЕЛЬНО
  computeCpgMix(P); 

  // 6. Равновесие (IAST)
  // Примесей нет -> Yi примесей = 0 -> Qeq = 0.
  // Несущий газ (если он есть в изотерме) посчитается.
  for (size_t i = 0; i < Ngrid + 1; ++i)
  {
    double sum = 0.0;
    for (size_t j = 0; j < Ncomp; ++j) {
      Yi[j] = P[i * Ncomp + j] / Pt[i];
      sum += Yi[j];
    }
    // Нормализация (на случай ошибок округления double)
    //if (sum > 1e-20) { for (size_t j = 0; j < Ncomp; ++j) Yi[j] /= sum; }

    iastPerformance += mixture.predictMixture(1, Yi, Pt[i], Xi, Ni, &cachedP0[i * Ncomp * maxIsothermTerms],
                                              &cachedPsi[i * maxIsothermTerms], Tgs[i]);

    for (size_t j = 0; j < Ncomp; ++j) {
      Qeq[i * Ncomp + j] = Ni[j];
      Qeqnew[i * Ncomp + j] = Ni[j];
    }
  }

  // Qeq1 (Второй слой)
  for (size_t i = 0; i < Ngrid + 1; ++i)
  {
    double sum1 = 0.0;
    for (size_t j = 0; j < Ncomp; ++j) {
      Yi1[j] = P[i * Ncomp + j] / Pt[i];
      sum1 += Yi1[j];
    }
    //if (sum1 > 1e-20) { for (size_t j = 0; j < Ncomp; ++j) Yi1[j] /= sum1; }

    iastPerformance1 += mixture.predictMixture(0, Yi1, Pt[i], Xi1, Ni1, &cachedP01[i * Ncomp * maxIsothermTerms1],   
                                              &cachedPsi1[i * maxIsothermTerms1], Tgs[i]);

    for (size_t j = 0; j < Ncomp; ++j) {
      Qeq1[i * Ncomp + j] = Ni1[j];
      Qeqnew1[i * Ncomp + j] = Ni1[j];
    }
  }
}

void Pressurization::run()
{

  std::vector<std::ofstream> streams;
  std::ofstream movieStream;

    // create the output files
  //std::vector<std::ofstream> streams;
  for (size_t i = 0; i < Ncomp; i++)
  {
    std::string fileName = "component_" + std::to_string(i) + "_" + components[i].name + ".data";
    streams.emplace_back(std::ofstream{fileName});
  }

  movieStream.open("column.data");

  size_t column_nr = 1;
  movieStream << "# column " << column_nr++ << ": z  (column position)" << std::endl;
  movieStream << "# column " << column_nr++ << ": V  (velocity)" << std::endl;
  movieStream << "# column " << column_nr++ << ": Pt (total pressure)" << std::endl;
  for (size_t j = 0; j < Ncomp; ++j)
  {
    movieStream << "# column " << column_nr++ << ": component " << j << " Q     (loading) " << std::endl;
    movieStream << "# column " << column_nr++ << ": component " << j << " Qeq   (equilibrium loading)" << std::endl;
    movieStream << "# column " << column_nr++ << ": component " << j << " P     (partial pressure)" << std::endl;
    movieStream << "# column " << column_nr++ << ": component " << j << " Pnorm (normalized partial pressure)"
                << std::endl;
    movieStream << "# column " << column_nr++ << ": component " << j << " Dpdt  (derivative P with t)" << std::endl;
    movieStream << "# column " << column_nr++ << ": component " << j << " Dqdt  (derivative Q with t)" << std::endl;
  }

  movieStream << "# column " << column_nr++ << ": Tgs (Gas and solid Temperature)" << std::endl;
  movieStream << "# column " << column_nr++ << ": Tw (Wall Temperature)" << std::endl;
  movieStream << "# column " << column_nr++ << ": DTgsdt (derivative Tgs with t)" << std::endl;
  movieStream << "# column " << column_nr++ << ": DTWalldt (derivative Tw with t)" << std::endl;
  

  for (size_t step = 0; (step < Nsteps || autoSteps); ++step)
  {
    // compute new step
    computeStep(step);
    //std::cout << "step" << std::endl;

    double t = static_cast<double>(step) * dt;

    if (step % writeEvery == 0)
    {
      // write breakthrough output to files
      // column 1: dimensionless time
      // column 2: time [minutes]
      // column 3: normalized partial pressure
      movieStream.open("column.data");
      
      for (size_t j = 0; j < Ncomp; ++j)
      {
        streams[j] << t * v_in / L << " " << t / 60.0 << " "
                    << P[Ngrid * Ncomp + j] / ((Pt[Ngrid]) * components[j].Yi0) << std::endl;
      }

      for (size_t i = 0; i < Ngrid + 1; ++i)
      {
        movieStream << static_cast<double>(i) * dx << " ";
        movieStream << V[i] << " ";
        movieStream << Pt[i] << " ";
        if ( indexLeft == 0 ) {
          for (size_t j = 0; j < Ncomp; ++j)
        {
          movieStream << Q[i * Ncomp + j] << " " << Qeq[i * Ncomp + j] << " " << P[i * Ncomp + j] << " "
                      << P[i * Ncomp + j] / (Pt[i] * components[j].Yi0) << " " << Dpdt[i * Ncomp + j] << " "
                      << Dqdt[i * Ncomp + j] << " ";
        }
        }

        if ( i <= indexLeft && indexLeft != 0) {
          for (size_t j = 0; j < Ncomp; ++j)
        {
          movieStream << Q[i * Ncomp + j] << " " << Qeq1[i * Ncomp + j] << " " << P[i * Ncomp + j] << " "
                      << P[i * Ncomp + j] / (Pt[i] * components[j].Yi0) << " " << Dpdt[i * Ncomp + j] << " "
                      << Dqdt[i * Ncomp + j] << " ";
        }
        }

        if ( i >= indexRight && indexLeft != 0) {
          for (size_t j = 0; j < Ncomp; ++j)
        {
          movieStream << Q[i * Ncomp + j] << " " << Qeq[i * Ncomp + j] << " " << P[i * Ncomp + j] << " "
                      << P[i * Ncomp + j] / (Pt[i] * components[j].Yi0) << " " << Dpdt[i * Ncomp + j] << " "
                      << Dqdt[i * Ncomp + j] << " ";
        }
        }

        movieStream << Tgs[i] << " " << Tw[i] << " " << DTdt[i] << " " << DTdtWall[i] << " ";
        
        movieStream << "\n";
      }
      movieStream << "\n\n";
      
    }


    if (step % printEvery == 0)
    {
      size_t mid_index = Tgsnew.size() / 2;
      std::cout << "Timestep " + std::to_string(step) + ", time: " + std::to_string(t) + " [s]" << std::endl;
      std::cout << "    Average number of mixture-prediction steps: " +
                       std::to_string(static_cast<double>(iastPerformance.first) /
                                      static_cast<double>(iastPerformance.second))
                << std::endl;
      std::cout << "Current middle-point Tempreture: " + std::to_string(Tgsnew[mid_index]) << " K" << std::endl;
    }
  }

  std::cout << "Final timestep " + std::to_string(Nsteps) +
                   ", time: " + std::to_string(dt * static_cast<double>(Nsteps)) + " [s]"
            << std::endl;
  
}

void Pressurization::run(std::vector<std::ofstream>& extStreams, std::ofstream& extMovieStream)
{

  for (size_t step = 0; (step < Nsteps || autoSteps); ++step)
  {
    // compute new step
    computeStep(step);
    //std::cout << "step" << std::endl;

    double t = static_cast<double>(step) * dt;

    if (step % writeEvery == 0)
    {
      for (size_t j = 0; j < Ncomp; ++j)
      {
        extStreams[j] << t * v_in / L << " " << t / 60.0 << " "
                    << P[Ngrid * Ncomp + j] / ((Pt[Ngrid]) * components[j].Yi0) << std::endl;
      }

      for (size_t i = 0; i < Ngrid + 1; ++i)
      {
        extMovieStream << static_cast<double>(i) * dx << " ";
        extMovieStream << V[i] << " ";
        extMovieStream << Pt[i] << " ";
        if ( indexLeft == 0 ) {
          for (size_t j = 0; j < Ncomp; ++j)
        {
          extMovieStream << Q[i * Ncomp + j] << " " << Qeq[i * Ncomp + j] << " " << P[i * Ncomp + j] << " "
                      << P[i * Ncomp + j] / (Pt[i] * components[j].Yi0) << " " << Dpdt[i * Ncomp + j] << " "
                      << Dqdt[i * Ncomp + j] << " ";
        }
        }

        if ( i <= indexLeft && indexLeft != 0) {
          for (size_t j = 0; j < Ncomp; ++j)
        {
          extMovieStream << Q[i * Ncomp + j] << " " << Qeq1[i * Ncomp + j] << " " << P[i * Ncomp + j] << " "
                      << P[i * Ncomp + j] / (Pt[i] * components[j].Yi0) << " " << Dpdt[i * Ncomp + j] << " "
                      << Dqdt[i * Ncomp + j] << " ";
        }
        }

        if ( i >= indexRight && indexLeft != 0) {
          for (size_t j = 0; j < Ncomp; ++j)
        {
          extMovieStream << Q[i * Ncomp + j] << " " << Qeq[i * Ncomp + j] << " " << P[i * Ncomp + j] << " "
                      << P[i * Ncomp + j] / (Pt[i] * components[j].Yi0) << " " << Dpdt[i * Ncomp + j] << " "
                      << Dqdt[i * Ncomp + j] << " ";
        }
        }

        extMovieStream << Tgs[i] << " " << Tw[i] << " " << DTdt[i] << " " << DTdtWall[i] << " ";
        
        extMovieStream << "\n";
      }
      extMovieStream << "\n\n";
      
    }


    if (step % printEvery == 0)
    {
      size_t mid_index = Tgsnew.size() / 2;
      std::cout << "Timestep " + std::to_string(step) + ", time: " + std::to_string(t) + " [s]" << std::endl;
      std::cout << "    Average number of mixture-prediction steps: " +
                       std::to_string(static_cast<double>(iastPerformance.first) /
                                      static_cast<double>(iastPerformance.second))
                << std::endl;
      std::cout << "Current middle-point Tempreture: " + std::to_string(Tgsnew[mid_index]) << " K" << std::endl;
    }
  }

  std::cout << "Final timestep " + std::to_string(Nsteps) +
                   ", time: " + std::to_string(dt * static_cast<double>(Nsteps)) + " [s]"
            << std::endl;
  
}

#ifdef PYBUILD

py::array_t<double> Pressurization::compute()
{
  size_t colsize = 6 * Ncomp + 5;
  std::vector<std::vector<std::vector<double>>> brk;

  // loop can quit early if autoSteps
  for (size_t step = 0; (step < Nsteps || autoSteps); ++step)
  {
    // check for error from python side (keyboard interrupt)
    if (PyErr_CheckSignals() != 0)
    {
      throw py::error_already_set();
    }

    computeStep(step);
    double t = static_cast<double>(step) * dt;
    if (step % writeEvery == 0)
    {
      std::vector<std::vector<double>> t_brk(Ngrid + 1, std::vector<double>(colsize));
      for (size_t i = 0; i < Ngrid + 1; ++i)
      {
        t_brk[i][0] = t * v_in / L;
        t_brk[i][1] = t / 60.0;
        t_brk[i][2] = static_cast<double>(i) * dx;
        t_brk[i][3] = V[i];
        t_brk[i][4] = Pt[i];

        for (size_t j = 0; j < Ncomp; ++j)
        {
          t_brk[i][5 + 6 * j] = Q[i * Ncomp + j];
          t_brk[i][6 + 6 * j] = Qeq[i * Ncomp + j];
          t_brk[i][7 + 6 * j] = P[i * Ncomp + j];
          t_brk[i][8 + 6 * j] = P[i * Ncomp + j] / (Pt[i] * components[j].Yi0);
          t_brk[i][9 + 6 * j] = Dpdt[i * Ncomp + j];
          t_brk[i][10 + 6 * j] = Dqdt[i * Ncomp + j];
        }
      }
      brk.push_back(t_brk);
    }
    if (step % printEvery == 0)
    {
      std::cout << "Timestep " + std::to_string(step) + ", time: " + std::to_string(t) + " [s]" << std::endl;
      std::cout << "    Average number of mixture-prediction steps: " +
                       std::to_string(static_cast<double>(iastPerformance.first) /
                                      static_cast<double>(iastPerformance.second))
                << std::endl;
    }
  }
  std::cout << "Final timestep " + std::to_string(Nsteps) +
                   ", time: " + std::to_string(dt * static_cast<double>(Nsteps)) + " [s]"
            << std::endl;

  std::vector<double> buffer;
  buffer.reserve(brk.size() * (Ngrid + 1) * colsize);
  for (const auto &vec1 : brk)
  {
    for (const auto &vec2 : vec1)
    {
      buffer.insert(buffer.end(), vec2.begin(), vec2.end());
    }
  }
  std::array<size_t, 3> shape{{brk.size(), Ngrid + 1, colsize}};
  py::array_t<double> py_breakthrough(shape, buffer.data());
  py_breakthrough.resize(shape);

  return py_breakthrough;
}

void Pressurization::setComponentsParameters(std::vector<double> molfracs, std::vector<double> params)
{
  size_t index = 0;
  for (size_t i = 0; i < Ncomp; ++i)
  {
    components[i].Yi0 = molfracs[i];
    size_t n_params = components[i].isotherm.numberOfParameters;
    std::vector<double> slicedVec(params.begin() + index, params.begin() + index + n_params);
    index = index + n_params;
    components[i].isotherm.setParameters(slicedVec);
  }

  // also set for mixture
  mixture.setComponentsParameters(molfracs, params);
}

std::vector<double> Breakthrough::getComponentsParameters()
{
  std::vector<double> params;
  for (size_t i = 0; i < Ncomp; ++i)
  {
    std::vector<double> compParams = components[i].isotherm.getParameters();
    params.insert(params.end(), compParams.begin(), compParams.end());
  }
  return params;
}
#endif  // PYBUILD

void Pressurization::computeStep(size_t step)
{
    double t = static_cast<double>(step) * dt;

    // ------------------------------------------------------------------
    // 0. ПРОВЕРКА СХОДИМОСТИ (Оставляем как было)
    // ------------------------------------------------------------------
    if (autoSteps)
    {
        double tolerance = 0.0;
        for (size_t j = 0; j < Ncomp; ++j) {
            tolerance = std::max(tolerance, std::abs((P[Ngrid * Ncomp + j] / (Pt[Ngrid] * components[j].Yi0)) - 1.0));
        }
        bool flag = false;

        for (size_t i=0; i < Ngrid + 1; i++) {
          if (Pt[i] >= p_total * 0.99) flag = true;
          else 
          {
            flag = false;
            break;
          }
        }

        //Pt[0] >= p_total * 0.9999 && Pt[Ngrid] >= p_total * 0.999

        if (flag) {
            std::cout << "\nPressure convergence reached!\n\n" << std::endl;
            Nsteps = static_cast<size_t>(1 * static_cast<double>(step));
            autoSteps = false;
        }
        if (tolerance < 0.01) {
            std::cout << "\nConvergence criteria reached\n\n" << std::endl;
            Nsteps = static_cast<size_t>(1.1 * static_cast<double>(step));
            autoSteps = false;
        }
    }

    // ------------------------------------------------------------------
    // 1. ГРАНИЧНЫЕ УСЛОВИЯ (RAMP)
    // ------------------------------------------------------------------
    // Рассчитываем целевое давление на текущий момент времени
    double target_pressure = p_total;
    double initial_pressure = Pmin;

    double current_P_in;
    if (t < ramp_time) {
        current_P_in = initial_pressure + (target_pressure - initial_pressure) * (t / ramp_time);
    } else {
        current_P_in = target_pressure;
    }

    // Применяем давление к граничной точке (i=0)
    // Обновляем И P (старый слой), И Pnew (новый слой), чтобы они были синхронны
    Pt[0] = 0.0;
    for (size_t j = 0; j < Ncomp; ++j)
    {
        double p_val = current_P_in * components[j].Yi0;
        
        P[0 * Ncomp + j] = p_val;
        Pnew[0 * Ncomp + j] = p_val; 
        
        Pt[0] += p_val;
    }

    // ------------------------------------------------------------------
    // 2. ОБНОВЛЕНИЕ ТЕРМОДИНАМИКИ (КРИТИЧЕСКИЙ МОМЕНТ)
    // ------------------------------------------------------------------
    // Мы изменили P[0]. Теперь нужно, чтобы Qeq[0] соответствовало этому давлению.
    // Вместо ручных циклов используем методы класса:
    
    // 1. Обновляем свойства (плотность, Cp...) по вектору P
    computeCpgMix(P);

    // 2. Считаем равновесие Qeqnew/Qeqnew1 на основе вектора Pnew.
    // Так как Pnew[0] мы только что задали, равновесие на входе пересчитается правильно.
    computeEquilibriumLoadings(); 

    // 3. КОПИРУЕМ рассчитанное равновесие в рабочие векторы Qeq/Qeq1.
    // Это нужно, чтобы на Step 1 функция производных видела правильное Qeq на входе.
    std::copy(Qeqnew.begin(), Qeqnew.end(), Qeq.begin());
    std::copy(Qeqnew1.begin(), Qeqnew1.end(), Qeq1.begin());

    // ==================================================================
    // SSP-RK Step 1
    // ==================================================================

    // Считаем производные. Теперь Qeq, P и V согласованы.
    computeFirstDerivatives(Dqdt, Dpdt, DTdt, DTdtWall, Qeq, Qeq1, Q, V, P, Tgs, Tw);

    // Euler Update
    // ВАЖНО: Начинаем с i=1. Точка 0 управляется блоком RAMP выше.
    for (size_t i = 0; i < Ngrid + 1; ++i)
    {
        // Сорбцию считаем везде (включая вход)
        Tgsnew[i] = Tgs[i] + dt * DTdt[i];
        Twnew[i] = Tw[i] + dt * DTdtWall[i]; 

        for (size_t j = 0; j < Ncomp; ++j) {
            Qnew[i * Ncomp + j] = Q[i * Ncomp + j] + dt * Dqdt[i * Ncomp + j];
            Pnew[i * Ncomp + j] = P[i * Ncomp + j] + dt * Dpdt[i * Ncomp + j];  
            }
        }

        // Газовые уравнения решаем только внутри колонны
        //if (i > 0) {
          //  for (size_t j = 0; j < Ncomp; ++j) {
            //    Pnew[i * Ncomp + j] = P[i * Ncomp + j] + dt * Dpdt[i * Ncomp + j];     
           // }
            //Tgsnew[i] = Tgs[i] + dt * DTdt[i];
            //Twnew[i] = Tw[i] + dt * DTdtWall[i];
        //}
    //}

    // Обновляем свойства для Pnew (которое изменилось на шаге Эйлера)
    computeCpgMix(Pnew);
    computeEquilibriumLoadings(); // Обновляет Qeqnew
    
    // Считаем скорость. ПЕРЕДАЕМ Dpdt (производные текущего шага).
    computeVelocityTemperatureLight(Dpdt, DTdt); 

    // ==================================================================
    // SSP-RK Step 2
    // ==================================================================

    computeFirstDerivatives(Dqdtnew, Dpdtnew, DTdtnew, DTdtWallnew, Qeqnew, Qeqnew1, Qnew, Vnew, Pnew, Tgsnew, Twnew);

    for (size_t i = 0; i < Ngrid + 1; ++i)
    { 
      Tgsnew[i] = 0.75 * Tgs[i] + 0.25 * Tgsnew[i] + 0.25 * dt * DTdtnew[i];
      Twnew[i] = 0.75 * Tw[i] + 0.25 * Twnew[i] + 0.25 * dt * DTdtWallnew[i];
        for (size_t j = 0; j < Ncomp; ++j) {
            Qnew[i * Ncomp + j] = 0.75 * Q[i * Ncomp + j] + 0.25 * Qnew[i * Ncomp + j] + 0.25 * dt * Dqdtnew[i * Ncomp + j];
            Pnew[i * Ncomp + j] = 0.75 * P[i * Ncomp + j] + 0.25 * Pnew[i * Ncomp + j] + 0.25 * dt * Dpdtnew[i * Ncomp + j];
        }
    }

    computeCpgMix(Pnew);
    computeEquilibriumLoadings();
    computeVelocityTemperatureLight(Dpdtnew, DTdtnew); // Передаем Dpdtnew

    // ==================================================================
    // SSP-RK Step 3
    // ==================================================================

    computeFirstDerivatives(Dqdtnew, Dpdtnew, DTdtnew, DTdtWallnew, Qeqnew, Qeqnew1, Qnew, Vnew, Pnew, Tgsnew, Twnew);

    for (size_t i = 0; i < Ngrid + 1; ++i)
    {
      Tgsnew[i] = (1.0 / 3.0) * Tgs[i] + (2.0 / 3.0) * Tgsnew[i] + (2.0 / 3.0) * dt * DTdtnew[i];
      Twnew[i] = (1.0 / 3.0) * Tw[i] + (2.0 / 3.0) * Twnew[i] + (2.0 / 3.0) * dt * DTdtWallnew[i];
        for (size_t j = 0; j < Ncomp; ++j) {
            Qnew[i * Ncomp + j] = (1.0 / 3.0) * Q[i * Ncomp + j] + (2.0 / 3.0) * Qnew[i * Ncomp + j] + (2.0 / 3.0) * dt * Dqdtnew[i * Ncomp + j];
            Pnew[i * Ncomp + j] = (1.0 / 3.0) * P[i * Ncomp + j] + (2.0 / 3.0) * Pnew[i * Ncomp + j] + (2.0 / 3.0) * dt * Dpdtnew[i * Ncomp + j];
        }
    }

    computeCpgMix(Pnew);
    computeEquilibriumLoadings();
    computeVelocityTemperatureLight(Dpdtnew, DTdtnew);

    // ==================================================================
    // FINAL UPDATE
    // ==================================================================
    std::copy(Qnew.begin(), Qnew.end(), Q.begin());
    std::copy(Pnew.begin(), Pnew.end(), P.begin());
    std::copy(Qeqnew.begin(), Qeqnew.end(), Qeq.begin());
    std::copy(Qeqnew1.begin(), Qeqnew1.end(), Qeq1.begin());
    std::copy(Vnew.begin(), Vnew.end(), V.begin());
    std::copy(Tgsnew.begin(), Tgsnew.end(), Tgs.begin());
    std::copy(Twnew.begin(), Twnew.end(), Tw.begin());
    
    // Блок Pulse Boundary Condition удален или закомментирован, 
    // так как он конфликтует с RAMP (мы уже задали P[0] выше).
}

void Pressurization::computeEquilibriumLoadings()
{
  // calculate new equilibrium loadings Qeqnew corresponding to the new timestep
  for (size_t i = 0; i < Ngrid + 1; ++i)
  {
    // estimation of total pressure Pt at each grid point from partial pressures
    Pt[i] = 0.0;
    for (size_t j = 0; j < Ncomp; ++j)
    {
      Pt[i] += std::max(0.0, Pnew[i * Ncomp + j]);
    }

    // compute gas-phase mol-fractions
    // force the gas-phase mol-fractions to be positive and normalized
    double sum = 0.0;
    for (size_t j = 0; j < Ncomp; ++j)
    {
      Yi[j] = std::max(Pnew[i * Ncomp + j], 0.0);
      sum += Yi[j];
    }
    for (size_t j = 0; j < Ncomp; ++j)
    {
      Yi[j] /= sum;
    }

    // use Yi and Pt[i] to compute the loadings in the adsorption mixture via mixture prediction
    iastPerformance += mixture.predictMixture(1, Yi, Pt[i], Xi, Ni, &cachedP0[i * Ncomp * maxIsothermTerms],
                                              &cachedPsi[i * maxIsothermTerms], Tgsnew[i]);

    for (size_t j = 0; j < Ncomp; ++j)
    {
      Qeqnew[i * Ncomp + j] = Ni[j];
    }
  }

  for (size_t i = 0; i < Ngrid + 1; ++i)
  {
    // estimation of total pressure Pt at each grid point from partial pressures
    Pt[i] = 0.0;
    for (size_t j = 0; j < Ncomp; ++j)
    {
      Pt[i] += std::max(0.0, Pnew[i * Ncomp + j]);
    }

    // compute gas-phase mol-fractions
    // force the gas-phase mol-fractions to be positive and normalized
    double sum1 = 0.0;
    for (size_t j = 0; j < Ncomp; ++j)
    {
      Yi1[j] = std::max(Pnew[i * Ncomp + j], 0.0);
      sum1 += Yi1[j];
    }
    for (size_t j = 0; j < Ncomp; ++j)
    {
      Yi1[j] /= sum1;
    }

    // use Yi and Pt[i] to compute the loadings in the adsorption mixture via mixture prediction
    iastPerformance1 += mixture.predictMixture(0, Yi1, Pt[i], Xi1, Ni1, &cachedP01[i * Ncomp * maxIsothermTerms1],  // clone
                                              &cachedPsi1[i * maxIsothermTerms1], Tgsnew[i]);

    for (size_t j = 0; j < Ncomp; ++j)
    {
      Qeqnew1[i * Ncomp + j] = Ni1[j];
    }
  }

  // check the total pressure at the outlet, it should not be negative
  if (Pt[0] + dptdx * L < 0.0)
  {
    throw std::runtime_error("Error: pressure gradient is too large (negative outlet pressure)\n");
  }
}

// calculate the derivatives Dq/dt and Dp/dt along the column
void Pressurization::computeFirstDerivatives(std::vector<double> &dqdt, std::vector<double> &dpdt, std::vector<double> &dTdt, std::vector<double> &dTdtWall,
                                           const std::vector<double> &q_eq, const std::vector<double> &q_eq1, const std::vector<double> &q,
                                           const std::vector<double> &v, const std::vector<double> &p, const std::vector<double> &Tmp, std::vector<double> &TmpWall)
{
    double idx = 1.0 / dx;
    double idx2 = 1.0 / (dx * dx);

    // =========================================================
    // 0. РАСЧЕТ ПРОИЗВОДНОЙ ДАВЛЕНИЯ НА ВХОДЕ (RAMP)
    // =========================================================
    
    // Считаем текущее давление на входе (сумма парциальных)
    double Pt_inlet = 0.0;
    for (size_t j = 0; j < Ncomp; ++j) {
        Pt_inlet += p[0 * Ncomp + j];
    }

    double ramp_duration = ramp_time; // Время открытия клапана
    double global_dPdt = 0.0;

    // Если текущее давление меньше целевого (с небольшим запасом на погрешность double),
    // значит мы все еще в стадии Ramp (подъема).
    // Производная линейной функции: (P_end - P_start) / time
    if (Pt_inlet < p_total * 0.99999) {
        global_dPdt = (p_total - Pmin) / ramp_duration;
    } else {
        // Мы вышли на плато, давление постоянно
        global_dPdt = 0.0;
    }

    // =========================================================
    // 1. ГРАНИЧНАЯ ТОЧКА (Вход i=0)
    // =========================================================
    if (indexLeft == 0) {
        for (size_t j = 0; j < Ncomp; ++j) {
            dqdt[0 * Ncomp + j] = components[j].Kl1 * (q_eq[0 * Ncomp + j] - q[0 * Ncomp + j]);
            
            // ЧЕСТНЫЙ РАСЧЕТ DPDT:
            // Парциальная производная = Yi0 * Полная производная
            dpdt[0 * Ncomp + j] = global_dPdt * components[j].Yi0;
            dTdt[0] = 0.0;             // T_in фиксировано
            dTdtWall[0] = 0.0;
        }
    } else {
        // Если вдруг вход попадает во второй слой (редкий кейс, но поддерживаем)
         for (size_t j = 0; j < Ncomp; ++j) {
            dqdt[0 * Ncomp + j] = components[j].Kl * (q_eq1[0 * Ncomp + j] - q[0 * Ncomp + j]);
            
            // ЧЕСТНЫЙ РАСЧЕТ DPDT:
            dpdt[0 * Ncomp + j] = global_dPdt * components[j].Yi0;
            
            dTdt[0] = 0.0;
            dTdtWall[0] = 0.0;
        }
    }

    // =========================================================
    // 2. ВНУТРЕННИЕ ТОЧКИ (Цикл по пространству)
    // =========================================================
    for (size_t i = 1; i < Ngrid; i++)
    {
        double sumH = 0.0;      
        double C_adsorbed_total = 0.0; 

        // -----------------------------------------------------
        // ШАГ A: Расчет dqdt и тепла адсорбции (sumH)
        // -----------------------------------------------------
        // Логика выбора слоя (Left / Interface / Right)
       
             // LAYER 1
        for (size_t j = 0; j < Ncomp; ++j) {
          double driving_force = (q_eq[i * Ncomp + j] - q[i * Ncomp + j]);
          dqdt[i * Ncomp + j] = components[j].Kl1 * driving_force;
          sumH += components[j].dH1 * components[j].Kl1 * driving_force;
          C_adsorbed_total += 0; 
        }

        for (size_t j = 0; j < Ncomp; ++j) {
            double driving_force = (q_eq[i * Ncomp + j] - q[i * Ncomp + j]);
            
            dpdt[i * Ncomp + j] = 
                  (v[i - 1] * p[(i - 1) * Ncomp + j] - v[i] * p[i * Ncomp + j]) * idx 
                + components[j].D1 * (p[(i + 1) * Ncomp + j] - 2.0 * p[i * Ncomp + j] + p[(i - 1) * Ncomp + j]) * idx2 
                - prefactorRight[j] * Tmp[i] * driving_force 
                + (v[i] * p[i * Ncomp + j] / Tmp[i]) * (Tmp[i] - Tmp[i - 1]) * idx;
                //+ (p[i * Ncomp + j] / Tmp[i]) * dTdt[i];
        }
        
        double dPtdt_local = 0;
        for (size_t j = 0; j < Ncomp; ++j) dPtdt_local += dpdt[i * Ncomp + j];
        

        double current_epsilon = epsilon1;
        double current_rhop = rho_p1;
        double current_Cps = Cps_1;
        double current_lambda_ax = lambda_ax_1;
        double current_h_in = h_in_1; 

        // -----------------------------------------------------
        // ШАГ B: Расчет dTdt (Температуры)
        // -----------------------------------------------------
        
        //double dVdz = (v[i] - v[i-1]) * idx; 
        //double WorkExpansion = Pt[i] * dVdz * 0; // Включаем, если Cv

        double WorkExpansion = dPtdt_local;

        double Numerator = current_lambda_ax * (Tmp[i + 1] - 2 * Tmp[i] + Tmp[i - 1]) * idx2 
                         - current_epsilon * (Pt[i] / (R * Tmp[i])) * Cpg_mix[i] * v[i] * (Tmp[i] - Tmp[i - 1]) * idx 
                         + (1.0 - current_epsilon) * current_rhop * sumH 
                         - 4.0 * current_h_in * (Tmp[i] - Tw[i]) / D_column_inner
                         + current_epsilon * WorkExpansion;

        double DenomTerm = (Pt[i] / (R * Tmp[i])) * current_epsilon * Cpg_mix[i] 
                         + (1.0 - current_epsilon) * current_rhop * (current_Cps + C_adsorbed_total);
        
        dTdt[i] = Numerator / DenomTerm;

        // -----------------------------------------------------
        // ШАГ C: Расчет dpdt (Давления) - ТЕПЕРЬ СРАЗУ
        // -----------------------
        for (size_t j = 0; j < Ncomp; ++j) dpdt[i * Ncomp + j] += (p[i * Ncomp + j] / Tmp[i]) * dTdt[i];

        // Стена
        dTdtWall[i] = (4 * D_column_inner * current_h_in * (Tmp[i] - TmpWall[i]) - 4 * D_column_out * h_out * (TmpWall[i] - Tamb)) / ((std::pow(D_column_out, 2) - std::pow(D_column_inner, 2)) * rho_wall * Cpw)
                    + lambda_w * (TmpWall[i + 1] - 2.0 * TmpWall[i] + TmpWall[i - 1]) * idx2 / (rho_wall * Cpw);
    }

    // =========================================================
    // 3. ПОСЛЕДНЯЯ ТОЧКА (i = Ngrid)
    // =========================================================
    {
        double sumH = 0.0;
        //double C_adsorbed_total = 0.0;

        // A. Сначала q и H (чтобы найти dTdt)
        for (size_t j = 0; j < Ncomp; ++j) {
             double driving_force = (q_eq[Ngrid * Ncomp + j] - q[Ngrid * Ncomp + j]);
             dqdt[Ngrid * Ncomp + j] = components[j].Kl1 * driving_force;
             sumH += components[j].dH1 * components[j].Kl1 * driving_force; 
        }

         for (size_t j = 0; j < Ncomp; ++j) {
            double driving_force = (q_eq[Ngrid * Ncomp + j] - q[Ngrid * Ncomp + j]);
            
            dpdt[Ngrid * Ncomp + j] = 
                  (v[Ngrid - 1] * p[(Ngrid - 1) * Ncomp + j] - v[Ngrid] * p[Ngrid * Ncomp + j]) * idx 
                + components[j].D1 * (p[(Ngrid - 1) * Ncomp + j] - p[Ngrid * Ncomp + j]) * idx2 
                - prefactorRight[j] * Tmp[Ngrid] * driving_force 
                + (v[Ngrid] * p[Ngrid * Ncomp + j] / Tmp[Ngrid]) * (Tmp[Ngrid] - Tmp[Ngrid - 1]) * idx;
                //+ (p[i * Ncomp + j] / Tmp[i]) * dTdt[i];
        }

        double dPtdt_local = 0;
        for (size_t j = 0; j < Ncomp; ++j) dPtdt_local += dpdt[Ngrid * Ncomp + j];
        
        // B. Считаем dTdt
        //double dVdz = (v[Ngrid] - v[Ngrid-1]) * idx * 0; 
        double WorkExpansion = dPtdt_local; 

        double Numerator = lambda_ax_1 * (Tmp[Ngrid - 1] - Tmp[Ngrid]) * idx2 
                         - (Pt[Ngrid] / (R * Tmp[Ngrid])) * Cpg_mix[Ngrid] * v[Ngrid] * (Tmp[Ngrid] - Tmp[Ngrid - 1]) * idx * 0 // dTdz = 0; Ngrid
                         + (1 - epsilon1) * rho_p1 * sumH 
                         - 4 * h_in_1 * (Tmp[Ngrid] - Tw[Ngrid]) / D_column_inner
                         + epsilon1 * WorkExpansion;

        double DenomTerm = (Pt[Ngrid] / (R * Tmp[Ngrid])) * epsilon1 * Cpg_mix[Ngrid] 
                         + (1 - epsilon1) * rho_p1 * Cps_1;
        
        dTdt[Ngrid] = Numerator / DenomTerm;

        for (size_t j = 0; j < Ncomp; ++j) dpdt[Ngrid * Ncomp + j] += (p[Ngrid * Ncomp + j] / Tmp[Ngrid]) * dTdt[Ngrid];

        dTdtWall[Ngrid] = (4 * D_column_inner * h_in_1 * (Tmp[Ngrid] - TmpWall[Ngrid]) - 4 * D_column_out * h_out * (TmpWall[Ngrid] - Tamb)) / ((std::pow(D_column_out, 2) - std::pow(D_column_inner, 2)) * rho_wall * Cpw)
                        + lambda_w * (TmpWall[Ngrid - 1] - TmpWall[Ngrid]) * idx2 / (rho_wall * Cpw);
    }
}

void Pressurization::computeFirstDerivatives2(
    std::vector<double> &dqdt, 
    std::vector<double> &dpdt, 
    std::vector<double> &dTdt, 
    std::vector<double> &dTdtWall,
    const std::vector<double> &q_eq, const std::vector<double> &q_eq1, const std::vector<double> &q,
    const std::vector<double> &v, const std::vector<double> &p, 
    const std::vector<double> &Tmp, std::vector<double> &TmpWall)
{
    double idx = 1.0 / dx;
    double idx2 = 1.0 / (dx * dx);

    // =========================================================
    // 0. РАСЧЕТ ПРОИЗВОДНОЙ ДАВЛЕНИЯ НА ВХОДЕ (RAMP)
    // =========================================================
    
    // Считаем текущее давление на входе (сумма парциальных)
    double Pt_inlet = 0.0;
    for (size_t j = 0; j < Ncomp; ++j) {
        Pt_inlet += p[0 * Ncomp + j];
    }

    double ramp_duration = ramp_time; // Время открытия клапана
    double global_dPdt = 0.0;

    // Если текущее давление меньше целевого (с небольшим запасом на погрешность double),
    // значит мы все еще в стадии Ramp (подъема).
    // Производная линейной функции: (P_end - P_start) / time
    if (Pt_inlet < p_total * 0.99999) {
        global_dPdt = (p_total - Pmin) / ramp_duration;
    } else {
        // Мы вышли на плато, давление постоянно
        global_dPdt = 0.0;
    }

    // =========================================================
    // 1. ГРАНИЧНАЯ ТОЧКА (Вход i=0)
    // =========================================================
    if (indexLeft == 0) {
        for (size_t j = 0; j < Ncomp; ++j) {
            dqdt[0 * Ncomp + j] = components[j].Kl1 * (q_eq[0 * Ncomp + j] - q[0 * Ncomp + j]);
            
            // ЧЕСТНЫЙ РАСЧЕТ DPDT:
            // Парциальная производная = Yi0 * Полная производная
            dpdt[0 * Ncomp + j] = global_dPdt * components[j].Yi0;
            dTdt[0] = 0.0;             // T_in фиксировано
            dTdtWall[0] = 0.0;
        }
    } else {
        // Если вдруг вход попадает во второй слой (редкий кейс, но поддерживаем)
         for (size_t j = 0; j < Ncomp; ++j) {
            dqdt[0 * Ncomp + j] = components[j].Kl * (q_eq1[0 * Ncomp + j] - q[0 * Ncomp + j]);
            
            // ЧЕСТНЫЙ РАСЧЕТ DPDT:
            dpdt[0 * Ncomp + j] = global_dPdt * components[j].Yi0;
            
            dTdt[0] = 0.0;
            dTdtWall[0] = 0.0;
        }
    }

    // =========================================================
    // 2. ВНУТРЕННИЕ ТОЧКИ (Цикл по пространству)
    // =========================================================
    for (size_t i = 1; i < Ngrid; i++)
    {
        double sumH = 0.0;      
        double C_adsorbed_total = 0.0; 

        // -----------------------------------------------------
        // ШАГ A: Расчет dqdt и тепла адсорбции (sumH)
        // -----------------------------------------------------
        // Логика выбора слоя (Left / Interface / Right)
        if (i <= indexLeft) { 
             // LAYER 0
             for (size_t j = 0; j < Ncomp; ++j) {
                double driving_force = (q_eq1[i * Ncomp + j] - q[i * Ncomp + j]);
                dqdt[i * Ncomp + j] = components[j].Kl * driving_force;
                sumH += components[j].dH * components[j].Kl * driving_force;
                C_adsorbed_total += 0; // q[...] * Cpg если нужно
             }
        } else {
             // LAYER 1
             for (size_t j = 0; j < Ncomp; ++j) {
                double driving_force = (q_eq[i * Ncomp + j] - q[i * Ncomp + j]);
                dqdt[i * Ncomp + j] = components[j].Kl1 * driving_force;
                sumH += components[j].dH1 * components[j].Kl1 * driving_force;
                C_adsorbed_total += 0; 
             }
        }

        double current_epsilon = (i <= indexLeft) ? epsilon : epsilon1;
        double current_rhop = (i <= indexLeft) ? rho_p : rho_p1;
        double current_Cps = (i <= indexLeft) ? Cps : Cps_1;
        double current_lambda_ax = (i <= indexLeft) ? lambda_ax : lambda_ax_1;
        double current_h_in = (i <= indexLeft) ? h_in : h_in_1; 

        // -----------------------------------------------------
        // ШАГ B: Расчет dTdt (Температуры)
        // -----------------------------------------------------
        
        double dVdz = (v[i] - v[i-1]) * idx; 
        double WorkExpansion = Pt[i] * dVdz; // Включаем, если Cv

        double Numerator = current_lambda_ax * (Tmp[i + 1] - 2 * Tmp[i] + Tmp[i - 1]) * idx2 
                         - current_epsilon * (Pt[i] / (R * Tmp[i])) * Cpg_mix[i] * v[i] * (Tmp[i] - Tmp[i - 1]) * idx 
                         + (1.0 - current_epsilon) * current_rhop * sumH 
                         - 4.0 * current_h_in * (Tmp[i] - Tw[i]) / D_column_inner
                         - current_epsilon * WorkExpansion;

        double DenomTerm = (Pt[i] / (R * Tmp[i])) * current_epsilon * Cpg_mix[i] 
                         + (1.0 - current_epsilon) * current_rhop * (current_Cps + C_adsorbed_total);
        
        dTdt[i] = Numerator / DenomTerm;

        // -----------------------------------------------------
        // ШАГ C: Расчет dpdt (Давления) - ТЕПЕРЬ СРАЗУ
        // -----------------------------------------------------
        
        if (i <= indexLeft) {
            // LAYER 0
            for (size_t j = 0; j < Ncomp; ++j) {
                double driving_force = (q_eq1[i * Ncomp + j] - q[i * Ncomp + j]);
                
                // Полная формула сразу:
                dpdt[i * Ncomp + j] = 
                      (v[i - 1] * p[(i - 1) * Ncomp + j] - v[i] * p[i * Ncomp + j]) * idx 
                    + components[j].D * (p[(i + 1) * Ncomp + j] - 2.0 * p[i * Ncomp + j] + p[(i - 1) * Ncomp + j]) * idx2 
                    - prefactorLeft[j] * Tmp[i] * driving_force 
                    + (v[i] * p[i * Ncomp + j] / Tmp[i]) * (Tmp[i] - Tmp[i - 1]) * idx
                    + (p[i * Ncomp + j] / Tmp[i]) * dTdt[i]; // <--- Добавили dTdt сразу
            }
        } else {
            // LAYER 1
            for (size_t j = 0; j < Ncomp; ++j) {
                double driving_force = (q_eq[i * Ncomp + j] - q[i * Ncomp + j]);
                
                dpdt[i * Ncomp + j] = 
                      (v[i - 1] * p[(i - 1) * Ncomp + j] - v[i] * p[i * Ncomp + j]) * idx 
                    + components[j].D1 * (p[(i + 1) * Ncomp + j] - 2.0 * p[i * Ncomp + j] + p[(i - 1) * Ncomp + j]) * idx2 
                    - prefactorRight[j] * Tmp[i] * driving_force 
                    + (v[i] * p[i * Ncomp + j] / Tmp[i]) * (Tmp[i] - Tmp[i - 1]) * idx
                    + (p[i * Ncomp + j] / Tmp[i]) * dTdt[i]; // <--- Добавили dTdt сразу
            }
        }

        // Стена
        dTdtWall[i] = (4 * D_column_inner * current_h_in * (Tmp[i] - TmpWall[i]) - 4 * D_column_out * h_out * (TmpWall[i] - Tamb)) / ((std::pow(D_column_out, 2) - std::pow(D_column_inner, 2)) * rho_wall * Cpw)
                    + lambda_w * (TmpWall[i + 1] - 2.0 * TmpWall[i] + TmpWall[i - 1]) * idx2 / (rho_wall * Cpw);
    }

    // =========================================================
    // 3. ПОСЛЕДНЯЯ ТОЧКА (i = Ngrid)
    // =========================================================
    {
        double sumH = 0.0;
        //double C_adsorbed_total = 0.0;

        // A. Сначала q и H (чтобы найти dTdt)
        for (size_t j = 0; j < Ncomp; ++j) {
             double driving_force = (q_eq[Ngrid * Ncomp + j] - q[Ngrid * Ncomp + j]);
             dqdt[Ngrid * Ncomp + j] = components[j].Kl1 * driving_force;
             sumH += components[j].dH1 * components[j].Kl1 * driving_force; 
        }

        // B. Считаем dTdt
        double dVdz = (v[Ngrid] - v[Ngrid-1]) * idx; 
        double WorkExpansion = Pt[Ngrid] * dVdz * 0; 

        double Numerator = lambda_ax_1 * (Tmp[Ngrid - 1] - Tmp[Ngrid]) * idx2 
                         - (Pt[Ngrid] / (R * Tmp[Ngrid])) * Cpg_mix[Ngrid] * v[Ngrid] * (Tmp[Ngrid] - Tmp[Ngrid - 1]) * idx 
                         + (1 - epsilon1) * rho_p1 * sumH 
                         - 4 * h_in_1 * (Tmp[Ngrid] - Tw[Ngrid]) / D_column_inner
                         - epsilon1 * WorkExpansion;

        double DenomTerm = (Pt[Ngrid] / (R * Tmp[Ngrid])) * epsilon1 * Cpg_mix[Ngrid] 
                         + (1 - epsilon1) * rho_p1 * Cps_1;
        
        dTdt[Ngrid] = Numerator / DenomTerm;

        // C. Считаем dpdt СРАЗУ
        for (size_t j = 0; j < Ncomp; ++j) {
             double driving_force = (q_eq[Ngrid * Ncomp + j] - q[Ngrid * Ncomp + j]);
             
             dpdt[Ngrid * Ncomp + j] = 
                  (v[Ngrid - 1] * p[(Ngrid - 1) * Ncomp + j] - v[Ngrid] * p[Ngrid * Ncomp + j]) * idx
                + components[j].D1 * (p[(Ngrid - 1) * Ncomp + j] - p[Ngrid * Ncomp + j]) * idx2 
                - prefactorRight[j] * Tmp[Ngrid] * driving_force
                + (p[Ngrid * Ncomp + j] / Tmp[Ngrid]) * dTdt[Ngrid]; // <--- dTdt тут уже известно
        }

        dTdtWall[Ngrid] = (4 * D_column_inner * h_in_1 * (Tmp[Ngrid] - TmpWall[Ngrid]) - 4 * D_column_out * h_out * (TmpWall[Ngrid] - Tamb)) / ((std::pow(D_column_out, 2) - std::pow(D_column_inner, 2)) * rho_wall * Cpw)
                        + lambda_w * (TmpWall[Ngrid - 1] - TmpWall[Ngrid]) * idx2 / (rho_wall * Cpw);
    }
}


std::vector<double> Pressurization::get_pressure() {
  return P;
}

std::vector<double> Pressurization::get_q() {
  return Q;
}

std::vector<double> Pressurization::get_T() {
  return Tgs;
}

void Pressurization::change_stage_status() {
  InitialStage = false; 
}

// calculate new velocity Vnew from Qnew, Qeqnew, Pnew, Pt
void Pressurization::computeVelocity()
{
  double idx2 = 1.0 / (dx * dx);

  // first grid point
  Vnew[0] = v_in;

  // middle gridpoints
  for (size_t i = 1; i < Ngrid; ++i)
  {
    // sum = derivative at the actual gridpoint i
    if (i < indexLeft) {
    //std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!INDEX LEFT DER!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    //std::cout << indexLeft << std::endl;
    double sum = 0.0;
    for (size_t j = 0; j < Ncomp; ++j)
    {
      sum =
          sum - prefactorLeft[j] * Tgsnew[i] * (Qeqnew1[i * Ncomp + j] - Qnew[i * Ncomp + j]) +
          components[j].D * (Pnew[(i - 1) * Ncomp + j] - 2.0 * Pnew[i * Ncomp + j] + Pnew[(i + 1) * Ncomp + j]) * idx2;
    }

    // explicit version
    Vnew[i] = Vnew[i - 1] + dx * (sum - Vnew[i - 1] * dptdx) / Pt[i];
  }

  if (i == indexLeft) {
    //std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!INDEX LEFT DER=!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    //std::cout << indexLeft << std::endl;
    double sum = 0.0;
    for (size_t j = 0; j < Ncomp; ++j)
    {
      sum =
          sum - prefactorLeftGP[j] * Tgsnew[i] * (Qeqnew1[i * Ncomp + j] - Qnew[i * Ncomp + j]) +
           (components[j].D * relLeft + components[j].D1 * relRight) * (Pnew[(i - 1) * Ncomp + j] - 2.0 * Pnew[i * Ncomp + j] + Pnew[(i + 1) * Ncomp + j]) * idx2;
    }

    // explicit version
    Vnew[i] = Vnew[i - 1] + dx * (sum - Vnew[i - 1] * dptdx) / Pt[i];
  }

  if (i == indexRight) {
    //std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!INDEX RIGHT DER=!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    //std::cout << indexRight << std::endl;
    double sum = 0.0;
    for (size_t j = 0; j < Ncomp; ++j)
    {
      sum =
          sum - prefactorRightGP[j] * Tgsnew[i] * (Qeqnew[i * Ncomp + j] - Qnew[i * Ncomp + j]) +
           (components[j].D * relRight + components[j].D1 * relLeft) * (Pnew[(i - 1) * Ncomp + j] - 2.0 * Pnew[i * Ncomp + j] + Pnew[(i + 1) * Ncomp + j]) * idx2;
    }

    // explicit version
    Vnew[i] = Vnew[i - 1] + dx * (sum - Vnew[i - 1] * dptdx) / Pt[i];
  }

  if (i > indexRight) {
    //std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!INDEX RIGHT DER!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    //std::cout << indexRight << std::endl;
    double sum = 0.0;
    for (size_t j = 0; j < Ncomp; ++j)
    {
      sum =
          sum - prefactorRight[j] * Tgsnew[i] * (Qeqnew[i * Ncomp + j] - Qnew[i * Ncomp + j]) +
           components[j].D1 * (Pnew[(i - 1) * Ncomp + j] - 2.0 * Pnew[i * Ncomp + j] + Pnew[(i + 1) * Ncomp + j]) * idx2;
    }

    // explicit version
    Vnew[i] = Vnew[i - 1] + dx * (sum - Vnew[i - 1] * dptdx) / Pt[i];
  }
    
  }

  // last grid point
  double sum = 0.0;
  for (size_t j = 0; j < Ncomp; ++j)
  {
    sum = sum - prefactorRight[j] * Tgsnew[Ngrid] * (Qeqnew[Ngrid * Ncomp + j] - Qnew[Ngrid * Ncomp + j]) +
          components[j].D1 * (Pnew[(Ngrid - 1) * Ncomp + j] - Pnew[Ngrid * Ncomp + j]) * idx2;
  }

  // explicit version
  Vnew[Ngrid] = Vnew[Ngrid - 1] + dx * (sum - Vnew[Ngrid - 1] * dptdx) / Pt[Ngrid];
}

void Pressurization::computeCpgMix(std::vector<double> &Pi) {
  //std::fill(Mol_mix.begin(), Mol_mix.end(), 0.0);
  std::fill(Cpg_mix.begin(), Cpg_mix.end(), 0.0);
  for (size_t i = 0; i < Ngrid + 1; ++i)
  {
    for (size_t j = 0; j < Ncomp; ++j)
    { 
       //Mol_mix[i] += components[j].MolMass * P[i * Ncomp + j] / Pt[i]; //  Mol mix calculating
       Cpg_mix[i] += components[j].Cpg * Pi[i * Ncomp + j] / Pt[i]; 
    }
  }  
}

void Pressurization::computeVelocityTemperatureLight(const std::vector<double>& current_dpdt, const std::vector<double>& current_dTdt)
{
  // 1. Граничное условие: Тупик на ПРАВОЙ границе.
    // Если Ngrid - это количество ячеек, то граней Ngrid + 1 (от 0 до Ngrid).
    // Vnew[Ngrid] - это скорость на глухой стенке.

    // 2. Интегрирование ОТ СТЕНКИ (Ngrid) КО ВХОДУ (0)
    // Используем цикл, который гарантированно проходит 0.
    // Логика: i - это индекс текущей ячейки.
    // Vnew[i+1] - скорость, выходящая из ячейки (справа), которую мы УЖЕ посчитали.
    // Vnew[i]   - скорость, входящая в ячейку (слева), которую ИЩЕМ.
    //Vnew[Ngrid] = 0;
    
    // Цикл пробежит значения i: Ngrid-1, Ngrid-2, ..., 1, 0
    for (size_t i = Ngrid; i-- > 0; )
    {
        double current_Pt = Pt[i + 1];
        if (current_Pt < 1.0) current_Pt = 1.0; 
        
        double current_T = Tgsnew[i + 1];

        double sum_sorption = 0.0;
        const double idx2 = 1.0 / (dx * dx);
        
        
        // --- A. Слагаемое сжатия (Accumulation: 1/P * dP/dt) ---
        double dPt_dt_local = 0.0;
        for (size_t j = 0; j < Ncomp; ++j) {
            dPt_dt_local += current_dpdt[i * Ncomp + j];
        }

        //if (i == Ngrid - 1) std::cout << dPt_dt_local << std::endl;

        for (size_t j = 0; j < Ncomp; ++j) {
                 // Здесь ваша логика для правой границы
                 sum_sorption += prefactorRight[j] * current_T * (Qeqnew[(i + 1) * Ncomp + j] - Qnew[(i + 1) * Ncomp + j]) +
                        components[j].D1 * (Pnew[(i) * Ncomp + j] - Pnew[(i + 1) * Ncomp + j]) * idx2; 
            }

        double term_Accumulation = (1.0 / current_Pt) * dPt_dt_local;

        double dPdz = (Pt[i + 1] - Pt[i]) / dx;

       //std::cout << sum_sorption << std::endl;

        double thermal_expansion = Vnew[i] * (Tgsnew[i + 1] - Tgsnew[i]) / (Tgsnew[i + 1] * dx);
        
        double total_demand = term_Accumulation + (1.0 / current_Pt) * Vnew[i + 1] * dPdz + (1.0 / current_Pt) * sum_sorption - thermal_expansion - (current_dTdt[i + 1] / Tgsnew[i + 1]);
        
        // Vnew[i+1] уже рассчитан на предыдущей итерации цикла (или это стена 0.0)
        Vnew[i] = Vnew[i + 1] + total_demand * dx;

        //std::cout << Vnew[i] << " " << i << " " << term_Accumulation << std::endl;
    }
        //Vnew[Ngrid - 1] = Vnew[Ngrid - 2];
}

void Pressurization::computeVelocityTemperature(const std::vector<double>& current_dpdt)
{
     // Цикл пробежит значения i: Ngrid-1, Ngrid-2, ..., 1, 0

    Vnew[0] = v_in;

    for (size_t i = 0; i < Ngrid + 1 ; i++)
    {
        double current_Pt = Pt[i];
        if (current_Pt < 1.0) current_Pt = 1.0; 
        
        double current_T = Tgsnew[i];

        double sum_sorption = 0.0;
        const double idx2 = 1.0 / (dx * dx);
        
        
        // --- A. Слагаемое сжатия (Accumulation: 1/P * dP/dt) ---
        double dPt_dt_local = 0.0;
        for (size_t j = 0; j < Ncomp; ++j) {
            dPt_dt_local += current_dpdt[i * Ncomp + j];
        }

        //if (i == Ngrid - 1) std::cout << dPt_dt_local << std::endl;

        for (size_t j = 0; j < Ncomp; ++j) {
                 // Здесь ваша логика для правой границы
                 sum_sorption += prefactorRight[j] * current_T * (Qeqnew[i * Ncomp + j] - Qnew[i * Ncomp + j]) +
                        components[j].D1 * (Pnew[(i - 1) * Ncomp + j] - Pnew[i * Ncomp + j]) * idx2; 
            }

        double term_Accumulation = (1.0 / current_Pt) * dPt_dt_local;

        double dPdz = (Pt[i + 1] - Pt[i]) / dx;

       //std::cout << sum_sorption << std::endl;
        
        double total_demand = term_Accumulation + (1.0 / current_Pt) * Vnew[i] * dPdz + (1.0 / current_Pt) * sum_sorption;
        
        // Vnew[i+1] уже рассчитан на предыдущей итерации цикла (или это стена 0.0)
        Vnew[i + 1] = Vnew[i] - total_demand * dx;
    }
}

void Pressurization::print() const { std::cout << repr(); }

std::string Pressurization::repr() const
{
  std::string s;
  s += "Column properties\n";
  s += "=======================================================\n";
  s += "Display-name:                           " + displayName + "\n";
  s += "Temperature:                           " + std::to_string(T) + " [K]\n";
  s += "RelRight:                              " + std::to_string(relRight) + " [K]\n";
  s += "RelLeft:                               " + std::to_string(relLeft) + " [K]\n";
  s += "NCOMP:                               " + std::to_string(Ncomp) + " [K]\n";
  s += "Column length:                         " + std::to_string(L) + " [m]\n";
  s += "Column void-fraction for left layer:                  " + std::to_string(epsilon) + " [-]\n";
  s += "Column void-fraction for left right:                  " + std::to_string(epsilon1) + " [-]\n";
  s += "Particle density:                      " + std::to_string(rho_p) + " [kg/m^3]\n";
  s += "2nd Particle density:                  " + std::to_string(rho_p1) + " [kg/m^3]\n";
  s += "X-Coord of the boundary:               " + std::to_string(boundaryCoordinate) + " [m]\n";
  s += "Total pressure:                        " + std::to_string(p_total) + " [Pa]\n";
  s += "Pressure gradient:                     " + std::to_string(dptdx) + " [Pa/m]\n";
  s += "Column entrance interstitial velocity: " + std::to_string(v_in) + " [m/s]\n";
  s += "Ramp Time for pressure: " + std::to_string(ramp_time) + " [s]\n";
  s += "Pressure minimum: " + std::to_string(Pmin) + " [Pa]\n";
  s += "\n\n";

  s += "Pressurezation settings\n";
  s += "=======================================================\n";
  s += "Number of time steps:          " + std::to_string(Nsteps) + "\n";
  s += "Print every step:              " + std::to_string(printEvery) + "\n";
  s += "Write data every step:         " + std::to_string(writeEvery) + "\n";
  s += "\n\n";

  s += "Integration details\n";
  s += "=======================================================\n";
  s += "Time step:                     " + std::to_string(dt) + " [s]\n";
  s += "Number of column grid points:  " + std::to_string(Ngrid) + "\n";
  s += "Column spacing:                " + std::to_string(dx) + " [m]\n";
  s += "\n\n";

  s += "Component data\n";
  s += "=======================================================\n";
  s += "maximum isotherm terms for 1st layer:        " + std::to_string(maxIsothermTerms) + "\n";
  s += "maximum isotherm terms for 2nd layer:        " + std::to_string(maxIsothermTerms1) + "\n";
  for (size_t i = 0; i < Ncomp; ++i)
  {
    s += components[i].repr();
    s += "\n";
  }
  return s;
}

void Pressurization::createPlotScript()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream stream_graphs("make_graphs.bat");
  stream_graphs << "set PATH=%PATH%;C:\\Program Files\\gnuplot\\bin;C:\\Program "
                   "Files\\ffmpeg-master-latest-win64-gpl\\bin;C:\\Program Files\\ffmpeg\\bin\n";
  stream_graphs << "gnuplot.exe plot_breakthrough\n";
#else
  std::ofstream stream_graphs("make_graphs");
  stream_graphs << "#!/bin/sh\n";
  stream_graphs << "cd -- \"$(dirname \"$0\")\"\n";
  stream_graphs << "gnuplot plot_breakthrough\n";
#endif

#if (__cplusplus >= 201703L)
  std::filesystem::path path{"make_graphs"};
  std::filesystem::permissions(path, std::filesystem::perms::owner_exec, std::filesystem::perm_options::add);
#else
  chmod("make_graphs", S_IRWXU);
#endif

  std::ofstream stream("plot_breakthrough");
  stream << "set encoding utf8\n";
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  stream << "set xlabel 'Dimensionless time, {/Arial-Italic τ}={/Arial-Italic tv/L} / [-]' font \"Arial,14\"\n";
  stream << "set ylabel 'Concentration exit gas, {/Arial-Italic c}_i/{/Arial-Italic c}_{i,0} / [-]' offset 0.0,0 font "
            "\"Arial,14\"\n";
  stream << "set key outside top center horizontal samplen 2.5 height 0.5 spacing 1.5 font 'Arial, 10'\n";
#else
  stream << "set xlabel 'Dimensionless time, {/Helvetica-Italic τ}={/Helvetica-Italic tv/L} / [-]' font "
            "\"Helvetica,18\"\n";
  stream << "set ylabel 'Concentration exit gas, {/Helvetica-Italic c}_i/{/Helvetica-Italic c}_{i,0} / [-]' offset "
            "0.0,0 font \"Helvetica,18\"\n";
  stream << "set key outside top center horizontal samplen 2.5 height 0.5 spacing 1.5 font 'Helvetica, 10'\n";
#endif
  stream << "set bmargin 4\n";
  stream << "set yrange[0:]\n";

  stream << "set key title '" << displayName << " {/:Italic T}=" << T << " K, {/:Italic p_t}=" << p_total * 1e-3
         << " kPa'\n";

  stream << "set output 'breakthrough_dimensionless.pdf'\n";
  stream << "set term pdf color solid\n";

  // colorscheme from book 'gnuplot in action', listing 12.7
  stream << "set linetype 1 pt 5 ps 1 lw 4 lc rgb '0xee0000'\n";
  stream << "set linetype 2 pt 7 ps 1 lw 4 lc rgb '0x008b00'\n";
  stream << "set linetype 3 pt 9 ps 1 lw 4 lc rgb '0x0000cd'\n";
  stream << "set linetype 4 pt 11 ps 1 lw 4 lc rgb '0xff3fb3'\n";
  stream << "set linetype 5 pt 13 ps 1 lw 4 lc rgb '0x00cdcd'\n";
  stream << "set linetype 6 pt 15 ps 1 lw 4 lc rgb '0xcd9b1d'\n";
  stream << "set linetype 7 pt  4 ps 1 lw 4 lc rgb '0x8968ed'\n";
  stream << "set linetype 8 pt  6 ps 1 lw 4 lc rgb '0x8b8b83'\n";
  stream << "set linetype 9 pt  8 ps 1 lw 4 lc rgb '0x00bb00'\n";
  stream << "set linetype 10 pt 10 ps 1 lw 4 lc rgb '0x1e90ff'\n";
  stream << "set linetype 11 pt 12 ps 1 lw 4 lc rgb '0x8b2500'\n";
  stream << "set linetype 12 pt 14 ps 1 lw 4 lc rgb '0x000000'\n";

  stream << "ev=1\n";
  stream << "plot \\\n";
  for (size_t i = 0; i < Ncomp; i++)
  {
    std::string fileName = "component_" + std::to_string(i) + "_" + components[i].name + ".data";
    stream << "    " << "\"" << fileName << "\"" << " us ($1):($3) every ev" << " title \"" << components[i].name
           << " (y_i=" << components[i].Yi0 << ")\""
           << " with li lt " << i + 1 << (i < Ncomp - 1 ? ",\\" : "") << "\n";
  }
  stream << "set output 'breakthrough.pdf'\n";
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  stream << "set xlabel 'Time, {/Arial-Italic t} / [min.]' font \"Arial,14\"\n";
#else
  stream << "set xlabel 'Time, {/Helvetica-Italic t} / [min.]' font \"Helvetica,18\"\n";
#endif
  stream << "plot \\\n";
  for (size_t i = 0; i < Ncomp; i++)
  {
    std::string fileName = "component_" + std::to_string(i) + "_" + components[i].name + ".data";
    stream << "    " << "\"" << fileName << "\"" << " us ($2):($3) every ev" << " title \"" << components[i].name
           << " (y_i=" << components[i].Yi0 << ")\""
           << " with li lt " << i + 1 << (i < Ncomp - 1 ? ",\\" : "") << "\n";
  }
}

void Pressurization::createMovieScripts()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movies.bat");
  makeMovieStream << "CALL make_movie_V.bat %1 %2 %3 %4\n";
  makeMovieStream << "CALL make_movie_T.bat %1 %2 %3 %4\n";
  makeMovieStream << "CALL make_movie_Tw.bat %1 %2 %3 %4\n";
  makeMovieStream << "CALL make_movie_Pt.bat %1 %2 %3 %4\n";
  makeMovieStream << "CALL make_movie_Q.bat %1 %2 %3 %4\n";
  makeMovieStream << "CALL make_movie_Qeq.bat %1 %2 %3 %4\n";
  makeMovieStream << "CALL make_movie_P.bat %1 %2 %3 %4\n";
  makeMovieStream << "CALL make_movie_Pnorm.bat %1 %2 %3 %4\n";
  makeMovieStream << "CALL make_movie_Dpdt.bat %1 %2 %3 %4\n";
  makeMovieStream << "CALL make_movie_Dqdt.bat %1 %2 %3 %4\n";
#else
  std::ofstream makeMovieStream("make_movies");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
  makeMovieStream << "./make_movie_V \"$@\"\n";
  makeMovieStream << "./make_movie_T \"$@\"\n";
  makeMovieStream << "./make_movie_Pt \"$@\"\n";
  makeMovieStream << "./make_movie_Q \"$@\"\n";
  makeMovieStream << "./make_movie_Qeq \"$@\"\n";
  makeMovieStream << "./make_movie_P \"$@\"\n";
  makeMovieStream << "./make_movie_Pnorm \"$@\"\n";
  makeMovieStream << "./make_movie_Dpdt \"$@\"\n";
  makeMovieStream << "./make_movie_Dqdt \"$@\"\n";
#endif

#if (__cplusplus >= 201703L)
  std::filesystem::path path{"make_movies"};
  std::filesystem::permissions(path, std::filesystem::perms::owner_exec, std::filesystem::perm_options::add);
#else
  chmod("make_movies", S_IRWXU);
#endif

  createMovieScriptColumnV();
  createMovieScriptColumnT();
  createMovieScriptColumnTw();
  createMovieScriptColumnPt();
  createMovieScriptColumnQ();
  createMovieScriptColumnQeq();
  createMovieScriptColumnP();
  createMovieScriptColumnDpdt();
  createMovieScriptColumnDqdt();
  createMovieScriptColumnPnormalized();
}

// -crf 18: the range of the CRF scale is 0–51, where 0 is lossless, 23 is the default,
//          and 51 is worst quality possible; 18 is visually lossless or nearly so.
// -pix_fmt yuv420p: needed on apple devices
std::string movieScriptTemplateP(std::string s)
{
  std::ostringstream stream;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  stream << "del column_movie_" << s << ".mp4\n";
  stream << "set /A argVec[1]=1\n";
  stream << "set /A argVec[2]=1200\n";
  stream << "set /A argVec[3]=800\n";
  stream << "set /A argVec[4]=18\n";
  stream << "setlocal enabledelayedexpansion\n";
  stream << "set argCount=0\n";
  stream << "for %%x in (%*) do (\n";
  stream << "   set /A argCount+=1\n";
  stream << "   set \"argVec[!argCount!]=%%~x\"'n";
  stream << ")\n";
  stream << "set PATH=%PATH%;C:\\Program Files\\gnuplot\\bin;C:\\Program "
            "Files\\ffmpeg-master-latest-win64-gpl\\bin;C:\\Program Files\\ffmpeg\\bin\n";
  stream << "gnuplot.exe -c plot_column_" << s
         << " %argVec[1]% %argVec[2]% %argVec[3]% | ffmpeg.exe -f png_pipe -s:v \"%argVec[2]%,%argVec[3]%\" -i pipe: "
            "-c:v libx264 -pix_fmt yuv420p -crf %argVec[4]% -c:a aac column_movie_"
         << s + ".mp4\n";
#else
  stream << "rm -f " << "column_movie_" << s << ".mp4\n";
  stream << "every=1\n";
  stream << "format=\"-c:v libx265 -tag:v hvc1\"\n";
  stream << "width=1200\n";
  stream << "height=800\n";
  stream << "quality=18\n";
  stream << "while getopts e:w:h:q:l flag\n";
  stream << "do\n";
  stream << "    case \"${flag}\" in\n";
  stream << "        e) every=${OPTARG};;\n";
  stream << "        w) width=${OPTARG};;\n";
  stream << "        h) height=${OPTARG};;\n";
  stream << "        q) quality=${OPTARG};;\n";
  stream << "        l) format=\"-c:v libx264\";;\n";
  stream << "    esac\n";
  stream << "done\n";
  stream << "gnuplot -c plot_column_" << s
         << " $every $width $height | ffmpeg -f png_pipe -s:v \"${width},${height}\" -i pipe: $format -pix_fmt yuv420p "
            "-crf $quality -c:a aac column_movie_"
         << s + ".mp4\n";
#endif
  return stream.str();
}

void Pressurization::createMovieScriptColumnV()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_V.bat");
#else
  std::ofstream makeMovieStream("make_movie_V");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateP("V");

#if (__cplusplus >= 201703L)
  std::filesystem::path path{"make_movie_V"};
  std::filesystem::permissions(path, std::filesystem::perms::owner_exec, std::filesystem::perm_options::add);
#else
  chmod("make_movie_V", S_IRWXU);
#endif

  std::ofstream stream("plot_column_V");

  stream << "set encoding utf8\n";
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  stream << "set terminal pngcairo size ARG2,ARG3 enhanced font 'Arial,10'\n";
  stream << "set xlabel 'Adsorber position / [m]' font 'Arial,14'\n";
  stream << "set ylabel 'Interstitial velocity, {/Arial-Italic v} / [m/s]' offset 0.0,0 font 'Arial,14'\n";
  stream << "set key outside top center horizontal samplen 2.5 height 0.5 spacing 1.5 font 'Arial, 10'\n";
#else
  stream << "set terminal pngcairo size ARG2,ARG3 enhanced font 'Helvetica,10'\n";
  stream << "set xlabel 'Adsorber position / [m]' font 'Helvetica,18'\n";
  stream << "set ylabel 'Interstitial velocity, {/Helvetica-Italic v} / [m/s]' offset 0.0,0 font 'Helvetica,18'\n";
  stream << "set key outside top center horizontal samplen 2.5 height 0.5 spacing 1.5 font 'Helvetica, 10'\n";
#endif

  // colorscheme from book 'gnuplot in action', listing 12.7
  stream << "set linetype 1 pt 5 ps 1 lw 4 lc rgb '0xee0000'\n";
  stream << "set linetype 2 pt 7 ps 1 lw 4 lc rgb '0x008b00'\n";
  stream << "set linetype 3 pt 9 ps 1 lw 4 lc rgb '0x0000cd'\n";
  stream << "set linetype 4 pt 11 ps 1 lw 4 lc rgb '0xff3fb3'\n";
  stream << "set linetype 5 pt 13 ps 1 lw 4 lc rgb '0x00cdcd'\n";
  stream << "set linetype 6 pt 15 ps 1 lw 4 lc rgb '0xcd9b1d'\n";
  stream << "set linetype 7 pt  4 ps 1 lw 4 lc rgb '0x8968ed'\n";
  stream << "set linetype 8 pt  6 ps 1 lw 4 lc rgb '0x8b8b83'\n";
  stream << "set linetype 9 pt  8 ps 1 lw 4 lc rgb '0x00bb00'\n";
  stream << "set linetype 10 pt 10 ps 1 lw 4 lc rgb '0x1e90ff'\n";
  stream << "set linetype 11 pt 12 ps 1 lw 4 lc rgb '0x8b2500'\n";
  stream << "set linetype 12 pt 14 ps 1 lw 4 lc rgb '0x000000'\n";

  stream << "set bmargin 4\n";
  stream << "set title '" << displayName << " {/:Italic T}=" << T << " K, {/:Italic p_t}=" << p_total * 1e-3
         << " kPa'\n";
  stream << "stats 'column.data' us 2 nooutput\n";
  stream << "max=STATS_max\n";
  stream << "stats 'column.data' us 1 nooutput\n";
  stream << "set xrange[0:STATS_max]\n";
  stream << "set yrange[0:1.1*max]\n";
  stream << "ev=int(ARG1)\n";
  stream << "do for [i=0:int((STATS_blocks-2)/ev)] {\n";
  stream << "  plot \\\n";
  stream << "    " << "'column.data'" << " us 1:2 index ev*i notitle with li lt 1,\\\n";
  stream << "    " << "'column.data'" << " us 1:2 index ev*i notitle with po lt 1\n";
  stream << "}\n";
}

void Pressurization::createMovieScriptColumnT()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_T.bat");
#else
  std::ofstream makeMovieStream("make_movie_T");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateP("T");

#if (__cplusplus >= 201703L)
  std::filesystem::path path{"make_movie_T"};
  std::filesystem::permissions(path, std::filesystem::perms::owner_exec, std::filesystem::perm_options::add);
#else
  chmod("make_movie_T", S_IRWXU);
#endif

  std::ofstream stream("plot_column_T");

  stream << "set encoding utf8\n";
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  stream << "set terminal pngcairo size ARG2,ARG3 enhanced font 'Arial,10'\n";
  stream << "set xlabel 'Adsorber position / [m]' font 'Arial,14'\n";
  stream << "set ylabel 'Gas temperature, {/Arial-Italic T} / [K]' offset 0.0,0 font 'Arial,14'\n";
  stream << "set key outside top center horizontal samplen 2.5 height 0.5 spacing 1.5 font 'Arial, 10'\n";
#else
  stream << "set terminal pngcairo size ARG2,ARG3 enhanced font 'Helvetica,10'\n";
  stream << "set xlabel 'Adsorber position / [m]' font 'Helvetica,18'\n";
  stream << "set ylabel 'Interstitial velocity, {/Helvetica-Italic T} / [K]' offset 0.0,0 font 'Helvetica,18'\n";
  stream << "set key outside top center horizontal samplen 2.5 height 0.5 spacing 1.5 font 'Helvetica, 10'\n";
#endif

  // colorscheme from book 'gnuplot in action', listing 12.7
  stream << "set linetype 1 pt 5 ps 1 lw 4 lc rgb '0xee0000'\n";
  stream << "set linetype 2 pt 7 ps 1 lw 4 lc rgb '0x008b00'\n";
  stream << "set linetype 3 pt 9 ps 1 lw 4 lc rgb '0x0000cd'\n";
  stream << "set linetype 4 pt 11 ps 1 lw 4 lc rgb '0xff3fb3'\n";
  stream << "set linetype 5 pt 13 ps 1 lw 4 lc rgb '0x00cdcd'\n";
  stream << "set linetype 6 pt 15 ps 1 lw 4 lc rgb '0xcd9b1d'\n";
  stream << "set linetype 7 pt  4 ps 1 lw 4 lc rgb '0x8968ed'\n";
  stream << "set linetype 8 pt  6 ps 1 lw 4 lc rgb '0x8b8b83'\n";
  stream << "set linetype 9 pt  8 ps 1 lw 4 lc rgb '0x00bb00'\n";
  stream << "set linetype 10 pt 10 ps 1 lw 4 lc rgb '0x1e90ff'\n";
  stream << "set linetype 11 pt 12 ps 1 lw 4 lc rgb '0x8b2500'\n";
  stream << "set linetype 12 pt 14 ps 1 lw 4 lc rgb '0x000000'\n";

  stream << "set bmargin 4\n";
  stream << "set title '" << displayName << " {/:Italic Tgs_0}=" << T << " K, {/:Italic p_t}=" << p_total * 1e-3
         << " kPa'\n";
  stream << "stats 'column.data' us "<< std::to_string(4 + 6 * Ncomp) <<" nooutput\n";
  stream << "max=STATS_max\n";
  stream << "stats 'column.data' us 1 nooutput\n";
  stream << "set xrange[0:STATS_max]\n";
  stream << "set yrange[0.6*max:1.2*max]\n";
  stream << "ev=int(ARG1)\n";
  stream << "do for [i=0:int((STATS_blocks-2)/ev)] {\n";
  stream << "  plot \\\n";
  stream << "    " << "'column.data'" << " us 1:" << std::to_string(4 + 6 * Ncomp) << " index ev*i notitle with li lt 1,\\\n";
  stream << "    " << "'column.data'" << " us 1:"<< std::to_string(4 + 6 * Ncomp) << " index ev*i notitle with po lt 1\n";
  stream << "}\n";
}

void Pressurization::createMovieScriptColumnTw()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_Tw.bat");
#else
  std::ofstream makeMovieStream("make_movie_Tw");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateP("Tw");

#if (__cplusplus >= 201703L)
  std::filesystem::path path{"make_movie_Tw"};
  std::filesystem::permissions(path, std::filesystem::perms::owner_exec, std::filesystem::perm_options::add);
#else
  chmod("make_movie_T", S_IRWXU);
#endif

  std::ofstream stream("plot_column_Tw");

  stream << "set encoding utf8\n";
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  stream << "set terminal pngcairo size ARG2,ARG3 enhanced font 'Arial,10'\n";
  stream << "set xlabel 'Adsorber position / [m]' font 'Arial,14'\n";
  stream << "set ylabel 'Wall temperature, {/Arial-Italic Tw} / [K]' offset 0.0,0 font 'Arial,14'\n";
  stream << "set key outside top center horizontal samplen 2.5 height 0.5 spacing 1.5 font 'Arial, 10'\n";
#else
  stream << "set terminal pngcairo size ARG2,ARG3 enhanced font 'Helvetica,10'\n";
  stream << "set xlabel 'Adsorber position / [m]' font 'Helvetica,18'\n";
  stream << "set ylabel 'Interstitial velocity, {/Helvetica-Italic T} / [K]' offset 0.0,0 font 'Helvetica,18'\n";
  stream << "set key outside top center horizontal samplen 2.5 height 0.5 spacing 1.5 font 'Helvetica, 10'\n";
#endif

  // colorscheme from book 'gnuplot in action', listing 12.7
  stream << "set linetype 1 pt 5 ps 1 lw 4 lc rgb '0xee0000'\n";
  stream << "set linetype 2 pt 7 ps 1 lw 4 lc rgb '0x008b00'\n";
  stream << "set linetype 3 pt 9 ps 1 lw 4 lc rgb '0x0000cd'\n";
  stream << "set linetype 4 pt 11 ps 1 lw 4 lc rgb '0xff3fb3'\n";
  stream << "set linetype 5 pt 13 ps 1 lw 4 lc rgb '0x00cdcd'\n";
  stream << "set linetype 6 pt 15 ps 1 lw 4 lc rgb '0xcd9b1d'\n";
  stream << "set linetype 7 pt  4 ps 1 lw 4 lc rgb '0x8968ed'\n";
  stream << "set linetype 8 pt  6 ps 1 lw 4 lc rgb '0x8b8b83'\n";
  stream << "set linetype 9 pt  8 ps 1 lw 4 lc rgb '0x00bb00'\n";
  stream << "set linetype 10 pt 10 ps 1 lw 4 lc rgb '0x1e90ff'\n";
  stream << "set linetype 11 pt 12 ps 1 lw 4 lc rgb '0x8b2500'\n";
  stream << "set linetype 12 pt 14 ps 1 lw 4 lc rgb '0x000000'\n";

  stream << "set bmargin 4\n";
  stream << "set title '" << displayName << " {/:Italic Tw_0}=" << Tamb << " K, {/:Italic p_t}=" << p_total * 1e-3
         << " kPa'\n";
  stream << "stats 'column.data' us "<< std::to_string(5 + 6 * Ncomp) <<" nooutput\n";
  stream << "max=STATS_max\n";
  stream << "stats 'column.data' us 1 nooutput\n";
  stream << "set xrange[0:STATS_max]\n";
  stream << "set yrange[0.6*max:1.1*max]\n";
  stream << "ev=int(ARG1)\n";
  stream << "do for [i=0:int((STATS_blocks-2)/ev)] {\n";
  stream << "  plot \\\n";
  //stream << "    " << "'column.data'" << " us 1:17 index ev*i notitle with li lt 1,\\\n";
  //stream << "    " << "'column.data'" << " us 1:17 index ev*i notitle with po lt 1\n";
  stream << "    " << "'column.data'" << " us 1:" << std::to_string(5 + 6 * Ncomp) << " index ev*i notitle with li lt 1,\\\n";
  stream << "    " << "'column.data'" << " us 1:" << std::to_string(5 + 6 * Ncomp) << " index ev*i notitle with po lt 1\n";
  stream << "}\n";
}

void Pressurization::createMovieScriptColumnPt()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_Pt.bat");
#else
  std::ofstream makeMovieStream("make_movie_Pt");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateP("Pt");

#if (__cplusplus >= 201703L)
  std::filesystem::path path{"make_movie_Pt"};
  std::filesystem::permissions(path, std::filesystem::perms::owner_exec, std::filesystem::perm_options::add);
#else
  chmod("make_movie_Pt", S_IRWXU);
#endif

  std::ofstream stream("plot_column_Pt");

  stream << "set encoding utf8\n";
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  stream << "set terminal pngcairo size ARG2,ARG3 enhanced font 'Arial,10'\n";
  stream << "set xlabel 'Adsorber position / [m]' font 'Arial,14'\n";
  stream << "set ylabel 'Total Pressure, {/Arial-Italic p_t} / [Pa]' offset 0.0,0 font 'Arial,14'\n";
  stream << "set key outside top center horizontal samplen 2.5 height 0.5 spacing 1.5 font 'Arial, 10'\n";
#else
  stream << "set terminal pngcairo size ARG2,ARG3 enhanced font 'Helvetica,10'\n";
  stream << "set xlabel 'Adsorber position / [m]' font 'Helvetica,18'\n";
  stream << "set ylabel 'Total Pressure, {/Helvetica-Italic p_t} / [Pa]' offset 0.0,0 font 'Helvetica,18'\n";
  stream << "set key outside top center horizontal samplen 2.5 height 0.5 spacing 1.5 font 'Helvetica, 10'\n";
#endif

  // colorscheme from book 'gnuplot in action', listing 12.7
  stream << "set linetype 1 pt 5 ps 1 lw 4 lc rgb '0xee0000'\n";
  stream << "set linetype 2 pt 7 ps 1 lw 4 lc rgb '0x008b00'\n";
  stream << "set linetype 3 pt 9 ps 1 lw 4 lc rgb '0x0000cd'\n";
  stream << "set linetype 4 pt 11 ps 1 lw 4 lc rgb '0xff3fb3'\n";
  stream << "set linetype 5 pt 13 ps 1 lw 4 lc rgb '0x00cdcd'\n";
  stream << "set linetype 6 pt 15 ps 1 lw 4 lc rgb '0xcd9b1d'\n";
  stream << "set linetype 7 pt  4 ps 1 lw 4 lc rgb '0x8968ed'\n";
  stream << "set linetype 8 pt  6 ps 1 lw 4 lc rgb '0x8b8b83'\n";
  stream << "set linetype 9 pt  8 ps 1 lw 4 lc rgb '0x00bb00'\n";
  stream << "set linetype 10 pt 10 ps 1 lw 4 lc rgb '0x1e90ff'\n";
  stream << "set linetype 11 pt 12 ps 1 lw 4 lc rgb '0x8b2500'\n";
  stream << "set linetype 12 pt 14 ps 1 lw 4 lc rgb '0x000000'\n";

  stream << "set bmargin 4\n";
  stream << "set title '" << displayName << " {/:Italic T}=" << T << " K, {/:Italic p_t}=" << p_total * 1e-3
         << " kPa'\n";
  stream << "stats 'column.data' us 3 nooutput\n";
  stream << "max=STATS_max\n";
  stream << "stats 'column.data' us 1 nooutput\n";
  stream << "set xrange[0:STATS_max]\n";
  stream << "set yrange[0:1.1*max]\n";
  stream << "ev=int(ARG1)\n";
  stream << "do for [i=0:int((STATS_blocks-2)/ev)] {\n";
  stream << "  plot \\\n";
  stream << "    " << "'column.data'" << " us 1:3 index ev*i notitle with li lt 1,\\\n";
  stream << "    " << "'column.data'" << " us 1:3 index ev*i notitle with po lt 1\n";
  stream << "}\n";
}

void Pressurization::createMovieScriptColumnQ()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_Q.bat");
#else
  std::ofstream makeMovieStream("make_movie_Q");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateP("Q");

#if (__cplusplus >= 201703L)
  std::filesystem::path path{"make_movie_Q"};
  std::filesystem::permissions(path, std::filesystem::perms::owner_exec, std::filesystem::perm_options::add);
#else
  chmod("make_movie_Q", S_IRWXU);
#endif

  std::ofstream stream("plot_column_Q");

  stream << "set encoding utf8\n";
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  stream << "set terminal pngcairo size ARG2,ARG3 enhanced font 'Arial,10'\n";
  stream << "set xlabel 'Adsorber position / [m]' font 'Arial,14'\n";
  stream << "set ylabel 'Concentration, {/Arial-Italic c}_i / [mol/kg]' offset 0.0,0 font 'Arial,14'\n";
  stream << "set key outside top center horizontal samplen 2.5 height 0.5 spacing 1.5 font 'Arial, 10'\n";
#else
  stream << "set terminal pngcairo size ARG2,ARG3 enhanced font 'Helvetica,10'\n";
  stream << "set xlabel 'Adsorber position / [m]' font 'Helvetica,18'\n";
  stream << "set ylabel 'Concentration, {/Helvetica-Italic c}_i / [mol/kg]' offset 0.0,0 font 'Helvetica,18'\n";
  stream << "set key outside top center horizontal samplen 2.5 height 0.5 spacing 1.5 font 'Helvetica, 10'\n";
#endif

  // colorscheme from book 'gnuplot in action', listing 12.7
  stream << "set linetype 1 pt 5 ps 1 lw 4 lc rgb '0xee0000'\n";
  stream << "set linetype 2 pt 7 ps 1 lw 4 lc rgb '0x008b00'\n";
  stream << "set linetype 3 pt 9 ps 1 lw 4 lc rgb '0x0000cd'\n";
  stream << "set linetype 4 pt 11 ps 1 lw 4 lc rgb '0xff3fb3'\n";
  stream << "set linetype 5 pt 13 ps 1 lw 4 lc rgb '0x00cdcd'\n";
  stream << "set linetype 6 pt 15 ps 1 lw 4 lc rgb '0xcd9b1d'\n";
  stream << "set linetype 7 pt  4 ps 1 lw 4 lc rgb '0x8968ed'\n";
  stream << "set linetype 8 pt  6 ps 1 lw 4 lc rgb '0x8b8b83'\n";
  stream << "set linetype 9 pt  8 ps 1 lw 4 lc rgb '0x00bb00'\n";
  stream << "set linetype 10 pt 10 ps 1 lw 4 lc rgb '0x1e90ff'\n";
  stream << "set linetype 11 pt 12 ps 1 lw 4 lc rgb '0x8b2500'\n";
  stream << "set linetype 12 pt 14 ps 1 lw 4 lc rgb '0x000000'\n";

  stream << "set bmargin 4\n";
  stream << "set key title '" << displayName << " {/:Italic T}=" << T << " K, {/:Italic p_t}=" << p_total * 1e-3
         << " kPa'\n";
  stream << "stats 'column.data' nooutput\n";
  stream << "max = 0.0;\n";
  for (size_t i = 0; i < Ncomp; i++) {
    int col = 4 + static_cast<int>(i) * 6;
    stream << "  stats 'column.data' us " << col << " nooutput\n";
    stream << "  if (max < STATS_max) { max = STATS_max; }\n";
  }
  stream << "stats 'column.data' us 1 nooutput\n";
  stream << "set xrange[0:STATS_max]\n";
  stream << "set yrange[0:1.1*max]\n";
  stream << "ev=int(ARG1)\n";
  stream << "do for [i=0:int((STATS_blocks-2)/ev)] {\n";
  stream << "  plot \\\n";
  for (size_t i = 0; i < Ncomp; i++)
  {
    stream << "    " << "'column.data'" << " us 1:" << std::to_string(4 + i * 6) << " index ev*i notitle "
           << " with li lt " << i + 1 << ",\\\n";
  }
  for (size_t i = 0; i < Ncomp; i++)
  {
    stream << "    " << "'column.data'" << " us 1:" << std::to_string(4 + i * 6) << " index ev*i title '"
           << components[i].name << " (y_i=" << components[i].Yi0 << ")'"
           << " with po lt " << i + 1 << (i < Ncomp - 1 ? ",\\" : "") << "\n";
  }
  stream << "}\n";
}

void Pressurization::createMovieScriptColumnQeq()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_Qeq.bat");
#else
  std::ofstream makeMovieStream("make_movie_Qeq");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateP("Qeq");

#if (__cplusplus >= 201703L)
  std::filesystem::path path{"make_movie_Qeq"};
  std::filesystem::permissions(path, std::filesystem::perms::owner_exec, std::filesystem::perm_options::add);
#else
  chmod("make_movie_Qeq", S_IRWXU);
#endif

  std::ofstream stream("plot_column_Qeq");

  stream << "set encoding utf8\n";
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  stream << "set terminal pngcairo size ARG2,ARG3 enhanced font 'Arial,10'\n";
  stream << "set xlabel 'Adsorber position / [m]' font 'Arial,14'\n";
  stream << "set ylabel 'Concentration, {/Arial-Italic c}_i / [mol/kg]' offset 0.0,0 font 'Arial,14'\n";
  stream << "set key outside top center horizontal samplen 2.5 height 0.5 spacing 1.5 font 'Arial, 10'\n";
#else
  stream << "set terminal pngcairo size ARG2,ARG3 enhanced font 'Helvetica,10'\n";
  stream << "set xlabel 'Adsorber position / [m]' font 'Helvetica,18'\n";
  stream << "set ylabel 'Concentration, {/Helvetica-Italic c}_i / [mol/kg]' offset 0.0,0 font 'Helvetica,18'\n";
  stream << "set key outside top center horizontal samplen 2.5 height 0.5 spacing 1.5 font 'Helvetica, 10'\n";
#endif

  // colorscheme from book 'gnuplot in action', listing 12.7
  stream << "set linetype 1 pt 5 ps 1 lw 4 lc rgb '0xee0000'\n";
  stream << "set linetype 2 pt 7 ps 1 lw 4 lc rgb '0x008b00'\n";
  stream << "set linetype 3 pt 9 ps 1 lw 4 lc rgb '0x0000cd'\n";
  stream << "set linetype 4 pt 11 ps 1 lw 4 lc rgb '0xff3fb3'\n";
  stream << "set linetype 5 pt 13 ps 1 lw 4 lc rgb '0x00cdcd'\n";
  stream << "set linetype 6 pt 15 ps 1 lw 4 lc rgb '0xcd9b1d'\n";
  stream << "set linetype 7 pt  4 ps 1 lw 4 lc rgb '0x8968ed'\n";
  stream << "set linetype 8 pt  6 ps 1 lw 4 lc rgb '0x8b8b83'\n";
  stream << "set linetype 9 pt  8 ps 1 lw 4 lc rgb '0x00bb00'\n";
  stream << "set linetype 10 pt 10 ps 1 lw 4 lc rgb '0x1e90ff'\n";
  stream << "set linetype 11 pt 12 ps 1 lw 4 lc rgb '0x8b2500'\n";
  stream << "set linetype 12 pt 14 ps 1 lw 4 lc rgb '0x000000'\n";

  stream << "set bmargin 4\n";
  stream << "set key title '" << displayName << " {/:Italic T}=" << T << " K, {/:Italic p_t}=" << p_total * 1e-3
         << " kPa'\n";
  stream << "stats 'column.data' nooutput\n";
  stream << "max = 0.0;\n";
  for (size_t i = 0; i < Ncomp; i++) {
    int col = 5 + static_cast<int>(i) * 6;
    stream << "  stats 'column.data' us " << col << " nooutput\n";
    stream << "  if (max < STATS_max) { max = STATS_max; }\n";
  }
  stream << "stats 'column.data' us 1 nooutput\n";
  stream << "set xrange[0:STATS_max]\n";
  stream << "set yrange[0:1.1*max]\n";
  stream << "ev=int(ARG1)\n";
  stream << "do for [i=0:int((STATS_blocks-2)/ev)] {\n";
  stream << "  plot \\\n";
  for (size_t i = 0; i < Ncomp; i++)
  {
    stream << "    " << "'column.data'" << " us 1:" << std::to_string(5 + i * 6) << " index ev*i notitle "
           << " with li lt " << i + 1 << ",\\\n";
  }
  for (size_t i = 0; i < Ncomp; i++)
  {
    stream << "    " << "'column.data'" << " us 1:" << std::to_string(5 + i * 6) << " index ev*i title '"
           << components[i].name << " (y_i=" << components[i].Yi0 << ")'"
           << " with po lt " << i + 1 << (i < Ncomp - 1 ? ",\\" : "") << "\n";
  }
  stream << "}\n";
}

void Pressurization::createMovieScriptColumnP()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_P.bat");
#else
  std::ofstream makeMovieStream("make_movie_P");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateP("P");

#if (__cplusplus >= 201703L)
  std::filesystem::path path{"make_movie_P"};
  std::filesystem::permissions(path, std::filesystem::perms::owner_exec, std::filesystem::perm_options::add);
#else
  chmod("make_movie_P", S_IRWXU);
#endif

  std::ofstream stream("plot_column_P");

  stream << "set encoding utf8\n";
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  stream << "set terminal pngcairo size ARG2,ARG3 enhanced font 'Arial,10'\n";
  stream << "set xlabel 'Adsorber position / [m]' font 'Arial,14'\n";
  stream << "set ylabel 'Partial pressure, {/Arial-Italic p}_i / [Pa]' offset 0.0,0 font 'Arial,14'\n";
  stream << "set key outside top center horizontal samplen 2.5 height 0.5 spacing 1.5 font 'Arial, 10'\n";
#else
  stream << "set terminal pngcairo size ARG2,ARG3 enhanced font 'Helvetica,10'\n";
  stream << "set xlabel 'Adsorber position / [m]' font 'Helvetica,18'\n";
  stream << "set ylabel 'Partial pressure, {/Helvetica-Italic p}_i / [Pa]' offset 0.0,0 font 'Helvetica,18'\n";
  stream << "set key outside top center horizontal samplen 2.5 height 0.5 spacing 1.5 font 'Helvetica, 10'\n";
#endif

  // colorscheme from book 'gnuplot in action', listing 12.7
  stream << "set linetype 1 pt 5 ps 1 lw 4 lc rgb '0xee0000'\n";
  stream << "set linetype 2 pt 7 ps 1 lw 4 lc rgb '0x008b00'\n";
  stream << "set linetype 3 pt 9 ps 1 lw 4 lc rgb '0x0000cd'\n";
  stream << "set linetype 4 pt 11 ps 1 lw 4 lc rgb '0xff3fb3'\n";
  stream << "set linetype 5 pt 13 ps 1 lw 4 lc rgb '0x00cdcd'\n";
  stream << "set linetype 6 pt 15 ps 1 lw 4 lc rgb '0xcd9b1d'\n";
  stream << "set linetype 7 pt  4 ps 1 lw 4 lc rgb '0x8968ed'\n";
  stream << "set linetype 8 pt  6 ps 1 lw 4 lc rgb '0x8b8b83'\n";
  stream << "set linetype 9 pt  8 ps 1 lw 4 lc rgb '0x00bb00'\n";
  stream << "set linetype 10 pt 10 ps 1 lw 4 lc rgb '0x1e90ff'\n";
  stream << "set linetype 11 pt 12 ps 1 lw 4 lc rgb '0x8b2500'\n";
  stream << "set linetype 12 pt 14 ps 1 lw 4 lc rgb '0x000000'\n";

  stream << "set bmargin 4\n";
  stream << "set key title '" << displayName << " {/:Italic T}=" << T << " K, {/:Italic p_t}=" << p_total * 1e-3
         << " kPa'\n";
  stream << "stats 'column.data' nooutput\n";
  stream << "max = 0.0;\n";
  stream << "do for [i=6:STATS_columns:6] {\n";
  stream << "  stats 'column.data' us i nooutput\n";
  stream << "  if (max<STATS_max) {\n";
  stream << "    max=STATS_max\n";
  stream << "  }\n";
  stream << "}\n";
  stream << "stats 'column.data' us 1 nooutput\n";
  stream << "set xrange[0:STATS_max]\n";
  stream << "set yrange[0:1.1*max]\n";
  stream << "ev=int(ARG1)\n";
  stream << "do for [i=0:int((STATS_blocks-2)/ev)] {\n";
  stream << "  plot \\\n";
  for (size_t i = 0; i < Ncomp; i++)
  {
    stream << "    " << "'column.data'" << " us 1:" << std::to_string(6 + i * 6) << " index ev*i notitle "
           << " with li lt " << i + 1 << ",\\\n";
  }
  for (size_t i = 0; i < Ncomp; i++)
  {
    stream << "    " << "'column.data'" << " us 1:" << std::to_string(6 + i * 6) << " index ev*i title '"
           << components[i].name << " (y_i=" << components[i].Yi0 << ")'"
           << " with po lt " << i + 1 << (i < Ncomp - 1 ? ",\\" : "") << "\n";
  }
  stream << "}\n";
}

void Pressurization::createMovieScriptColumnPnormalized()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_Pnorm.bat");
#else
  std::ofstream makeMovieStream("make_movie_Pnorm");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateP("Pnorm");

#if (__cplusplus >= 201703L)
  std::filesystem::path path{"make_movie_Pnorm"};
  std::filesystem::permissions(path, std::filesystem::perms::owner_exec, std::filesystem::perm_options::add);
#else
  chmod("make_movie_Pnorm", S_IRWXU);
#endif

  std::ofstream stream("plot_column_Pnorm");

  stream << "set encoding utf8\n";
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  stream << "set terminal pngcairo size ARG2,ARG3 enhanced font 'Arial,10'\n";
  stream << "set xlabel 'Adsorber position / [m]' font 'Arial,14'\n";
  stream << "set ylabel 'Partial pressure, {/Arial-Italic p}_i / [-]' offset 0.0,0 font 'Arial,14'\n";
  stream << "set key outside top center horizontal samplen 2.5 height 0.5 spacing 1.5 font 'Arial, 10'\n";
#else
  stream << "set terminal pngcairo size ARG2,ARG3 enhanced font 'Helvetica,10'\n";
  stream << "set xlabel 'Adsorber position / [m]' font 'Helvetica,18'\n";
  stream << "set ylabel 'Partial pressure, {/Helvetica-Italic p}_i / [-]' offset 0.0,0 font 'Helvetica,18'\n";
  stream << "set key outside top center horizontal samplen 2.5 height 0.5 spacing 1.5 font 'Helvetica, 10'\n";
#endif

  // colorscheme from book 'gnuplot in action', listing 12.7
  stream << "set linetype 1 pt 5 ps 1 lw 4 lc rgb '0xee0000'\n";
  stream << "set linetype 2 pt 7 ps 1 lw 4 lc rgb '0x008b00'\n";
  stream << "set linetype 3 pt 9 ps 1 lw 4 lc rgb '0x0000cd'\n";
  stream << "set linetype 4 pt 11 ps 1 lw 4 lc rgb '0xff3fb3'\n";
  stream << "set linetype 5 pt 13 ps 1 lw 4 lc rgb '0x00cdcd'\n";
  stream << "set linetype 6 pt 15 ps 1 lw 4 lc rgb '0xcd9b1d'\n";
  stream << "set linetype 7 pt  4 ps 1 lw 4 lc rgb '0x8968ed'\n";
  stream << "set linetype 8 pt  6 ps 1 lw 4 lc rgb '0x8b8b83'\n";
  stream << "set linetype 9 pt  8 ps 1 lw 4 lc rgb '0x00bb00'\n";
  stream << "set linetype 10 pt 10 ps 1 lw 4 lc rgb '0x1e90ff'\n";
  stream << "set linetype 11 pt 12 ps 1 lw 4 lc rgb '0x8b2500'\n";
  stream << "set linetype 12 pt 14 ps 1 lw 4 lc rgb '0x000000'\n";

  stream << "set bmargin 4\n";
  stream << "set key title '" << displayName << " {/:Italic T}=" << T << " K, {/:Italic p_t}=" << p_total * 1e-3
         << " kPa'\n";
  stream << "stats 'column.data' nooutput\n";
  stream << "max = 0.0;\n";
  stream << "do for [i=7:STATS_columns:6] {\n";
  stream << "  stats 'column.data' us i nooutput\n";
  stream << "  if (max<STATS_max) {\n";
  stream << "    max=STATS_max\n";
  stream << "  }\n";
  stream << "}\n";
  stream << "stats 'column.data' us 1 nooutput\n";
  stream << "set xrange[0:STATS_max]\n";
  stream << "set yrange[0:1.1*max]\n";
  stream << "ev=int(ARG1)\n";
  stream << "do for [i=0:int((STATS_blocks-2)/ev)] {\n";
  stream << "  plot \\\n";
  for (size_t i = 0; i < Ncomp; i++)
  {
    stream << "    " << "'column.data'" << " us 1:" << std::to_string(7 + i * 6) << " index ev*i notitle "
           << " with li lt " << i + 1 << ",\\\n";
  }
  for (size_t i = 0; i < Ncomp; i++)
  {
    stream << "    " << "'column.data'" << " us 1:" << std::to_string(7 + i * 6) << " index ev*i title '"
           << components[i].name << " (y_i=" << components[i].Yi0 << ")'"
           << " with po lt " << i + 1 << (i < Ncomp - 1 ? ",\\" : "") << "\n";
  }
  stream << "}\n";
}

void Pressurization::createMovieScriptColumnDpdt()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_Dpdt.bat");
#else
  std::ofstream makeMovieStream("make_movie_Dpdt");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateP("Dpdt");

#if (__cplusplus >= 201703L)
  std::filesystem::path path{"make_movie_Dpdt"};
  std::filesystem::permissions(path, std::filesystem::perms::owner_exec, std::filesystem::perm_options::add);
#else
  chmod("make_movie_Dpdt", S_IRWXU);
#endif

  std::ofstream stream("plot_column_Dpdt");

  stream << "set encoding utf8\n";
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  stream << "set terminal pngcairo size ARG2,ARG3 enhanced font 'Arial,10'\n";
  stream << "set xlabel 'Adsorber position / [m]' font 'Arial,14'\n";
  stream << "set ylabel 'Pressure derivative, {/Arial-Italic dp_/dt} / [Pa/s]' offset 0.0,0 font 'Arial,14'\n";
  stream << "set key outside top center horizontal samplen 2.5 height 0.5 spacing 1.5 font 'Arial, 10'\n";
#else
  stream << "set terminal pngcairo size ARG2,ARG3 enhanced font 'Helvetica,10'\n";
  stream << "set xlabel 'Adsorber position / [m]' font 'Helvetica,18'\n";
  stream << "set ylabel 'Pressure derivative, {/Helvetica-Italic dp_/dt} / [Pa/s]' offset 0.0,0 font 'Helvetica,18'\n";
  stream << "set key outside top center horizontal samplen 2.5 height 0.5 spacing 1.5 font 'Helvetica, 10'\n";
#endif

  // colorscheme from book 'gnuplot in action', listing 12.7
  stream << "set linetype 1 pt 5 ps 1 lw 4 lc rgb '0xee0000'\n";
  stream << "set linetype 2 pt 7 ps 1 lw 4 lc rgb '0x008b00'\n";
  stream << "set linetype 3 pt 9 ps 1 lw 4 lc rgb '0x0000cd'\n";
  stream << "set linetype 4 pt 11 ps 1 lw 4 lc rgb '0xff3fb3'\n";
  stream << "set linetype 5 pt 13 ps 1 lw 4 lc rgb '0x00cdcd'\n";
  stream << "set linetype 6 pt 15 ps 1 lw 4 lc rgb '0xcd9b1d'\n";
  stream << "set linetype 7 pt  4 ps 1 lw 4 lc rgb '0x8968ed'\n";
  stream << "set linetype 8 pt  6 ps 1 lw 4 lc rgb '0x8b8b83'\n";
  stream << "set linetype 9 pt  8 ps 1 lw 4 lc rgb '0x00bb00'\n";
  stream << "set linetype 10 pt 10 ps 1 lw 4 lc rgb '0x1e90ff'\n";
  stream << "set linetype 11 pt 12 ps 1 lw 4 lc rgb '0x8b2500'\n";
  stream << "set linetype 12 pt 14 ps 1 lw 4 lc rgb '0x000000'\n";

  stream << "set bmargin 4\n";
  stream << "set key title '" << displayName << " {/:Italic T}=" << T << " K, {/:Italic p_t}=" << p_total * 1e-3
         << " kPa'\n";
  stream << "stats 'column.data' nooutput\n";
  stream << "max = -1e10;\n";
  stream << "min = 1e10;\n";
  stream << "do for [i=8:STATS_columns:6] {\n";
  stream << "  stats 'column.data' us i nooutput\n";
  stream << "  if (STATS_max>max) {\n";
  stream << "    max=STATS_max\n";
  stream << "  }\n";
  stream << "  if (STATS_min<min) {\n";
  stream << "    min=STATS_min\n";
  stream << "  }\n";
  stream << "}\n";
  stream << "stats 'column.data' us 1 nooutput\n";
  stream << "set xrange[0:STATS_max]\n";
  stream << "set yrange[1.1*min:1.1*max]\n";
  stream << "ev=int(ARG1)\n";
  stream << "do for [i=0:int((STATS_blocks-2)/ev)] {\n";
  stream << "  plot \\\n";
  for (size_t i = 0; i < Ncomp; i++)
  {
    stream << "    " << "'column.data'" << " us 1:" << std::to_string(8 + i * 6) << " index ev*i notitle "
           << " with li lt " << i + 1 << ",\\\n";
  }
  for (size_t i = 0; i < Ncomp; i++)
  {
    stream << "    " << "'column.data'" << " us 1:" << std::to_string(8 + i * 6) << " index ev*i title '"
           << components[i].name << " (y_i=" << components[i].Yi0 << ")'"
           << " with po lt " << i + 1 << (i < Ncomp - 1 ? ",\\" : "") << "\n";
  }
  stream << "}\n";
}

void Pressurization::createMovieScriptColumnDqdt()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_Dqdt.bat");
#else
  std::ofstream makeMovieStream("make_movie_Dqdt");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateP("Dqdt");

#if (__cplusplus >= 201703L)
  std::filesystem::path path{"make_movie_Dqdt"};
  std::filesystem::permissions(path, std::filesystem::perms::owner_exec, std::filesystem::perm_options::add);
#else
  chmod("make_movie_Dqdt", S_IRWXU);
#endif

  std::ofstream stream("plot_column_Dqdt");

  stream << "set encoding utf8\n";
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  stream << "set terminal pngcairo size ARG2,ARG3 enhanced font 'Arial,10'\n";
  stream << "set xlabel 'Adsorber position / [m]' font 'Arial,14'\n";
  stream << "set ylabel 'Loading derivative, {/Arial-Italic dq_i/dt} / [mol/kg/s]' offset 0.0,0 font 'Arial,14'\n";
  stream << "set key outside top center horizontal samplen 2.5 height 0.5 spacing 1.5 font 'Arial, 10'\n";
#else
  stream << "set terminal pngcairo size ARG2,ARG3 enhanced font 'Helvetica,10'\n";
  stream << "set xlabel 'Adsorber position / [m]' font 'Helvetica,18'\n";
  stream
      << "set ylabel 'Loading derivative, {/Helvetica-Italic dq_i/dt} / [mol/kg/s]' offset 0.0,0 font 'Helvetica,18'\n";
  stream << "set key outside top center horizontal samplen 2.5 height 0.5 spacing 1.5 font 'Helvetica, 10'\n";
#endif

  // colorscheme from book 'gnuplot in action', listing 12.7
  stream << "set linetype 1 pt 5 ps 1 lw 4 lc rgb '0xee0000'\n";
  stream << "set linetype 2 pt 7 ps 1 lw 4 lc rgb '0x008b00'\n";
  stream << "set linetype 3 pt 9 ps 1 lw 4 lc rgb '0x0000cd'\n";
  stream << "set linetype 4 pt 11 ps 1 lw 4 lc rgb '0xff3fb3'\n";
  stream << "set linetype 5 pt 13 ps 1 lw 4 lc rgb '0x00cdcd'\n";
  stream << "set linetype 6 pt 15 ps 1 lw 4 lc rgb '0xcd9b1d'\n";
  stream << "set linetype 7 pt  4 ps 1 lw 4 lc rgb '0x8968ed'\n";
  stream << "set linetype 8 pt  6 ps 1 lw 4 lc rgb '0x8b8b83'\n";
  stream << "set linetype 9 pt  8 ps 1 lw 4 lc rgb '0x00bb00'\n";
  stream << "set linetype 10 pt 10 ps 1 lw 4 lc rgb '0x1e90ff'\n";
  stream << "set linetype 11 pt 12 ps 1 lw 4 lc rgb '0x8b2500'\n";
  stream << "set linetype 12 pt 14 ps 1 lw 4 lc rgb '0x000000'\n";

  stream << "set bmargin 4\n";
  stream << "set key title '" << displayName << " {/:Italic T}=" << T << " K, {/:Italic p_t}=" << p_total * 1e-3
         << " kPa'\n";
  stream << "stats 'column.data' nooutput\n";
  stream << "max = -1e10;\n";
  stream << "min = 1e10;\n";
  stream << "min = 10000000000000.0;\n";
  stream << "do for [i=9:STATS_columns:6] {\n";
  stream << "  stats 'column.data' us i nooutput\n";
  stream << "  if (STATS_max>max) {\n";
  stream << "    max=STATS_max\n";
  stream << "  }\n";
  stream << "  if (STATS_min<min) {\n";
  stream << "    min=STATS_min\n";
  stream << "  }\n";
  stream << "}\n";
  stream << "stats 'column.data' us 1 nooutput\n";
  stream << "set xrange[0:STATS_max]\n";
  stream << "set yrange[1.1*min:1.1*max]\n";
  stream << "ev=int(ARG1)\n";
  stream << "do for [i=0:int((STATS_blocks-2)/ev)] {\n";
  stream << "  plot \\\n";
  for (size_t i = 0; i < Ncomp; i++)
  {
    stream << "    " << "'column.data'" << " us 1:" << std::to_string(9 + i * 6) << " index ev*i notitle "
           << " with li lt " << i + 1 << ",\\\n";
  }
  for (size_t i = 0; i < Ncomp; i++)
  {
    stream << "    " << "'column.data'" << " us 1:" << std::to_string(9 + i * 6) << " index ev*i title '"
           << components[i].name << " (y_i=" << components[i].Yi0 << ")'"
           << " with po lt " << i + 1 << (i < Ncomp - 1 ? ",\\" : "") << "\n";
  }
  stream << "}\n";
}