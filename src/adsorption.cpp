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

#include "adsorption.h"
#include "DataReader.h"
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

Adsorption::Adsorption(const InputReader &inputReader)
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
      numberOfLayers(inputReader.numberOfLayers),
      timeStage(inputReader.TimeStage),
      v_in(inputReader.columnEntranceVelocityAds),
      L(inputReader.columnLength),
      dx(L / static_cast<double>(Ngrid)),
      dt(inputReader.timeStepAds),
      Nsteps(inputReader.numberOfTimeSteps),
      autoSteps(inputReader.autoNumberOfTimeSteps),
      pulse(inputReader.pulseBreakthrough),
      tpulse(inputReader.pulseTime),
      mixture(inputReader),
      mixture1(inputReader),
      maxIsothermTerms(inputReader.maxIsothermTerms),
      maxIsothermTerms1(inputReader.maxIsothermTerms1),
      prefactors((Ngrid + 1) * Ncomp),
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
      Bd(inputReader.BoundaryCoords),
      indexes(numberOfLayers + 1),
      rho_particle(inputReader.rho_particle),
      eps(inputReader.eps),
      Cps_layer(inputReader.Cps_layer),
      lambda_x_layer(inputReader.lambda_x_layer),
      h_in_layer(inputReader.h_in_layer),
      CarrierGasExistance(inputReader.CarrierGasExistance),
      IsothermalRegime(inputReader.IsothermalRegime),
      cycle(inputReader.Cycle)
{

  size_t temp_index = 0;

  indexes[0] = 0;
  indexes[numberOfLayers] = Ngrid + 1;

  Bd.resize(numberOfLayers - 1);

  for (size_t k = 0; k < numberOfLayers - 1; ++k) {
    for (size_t i = temp_index; i < Ngrid + 1; ++i) {
       if (Bd[k] < static_cast<double>(i) * dx) {
         indexes[k + 1] = i;
         temp_index = i;
         break;
       }
    }
  }

  for (size_t i = 0; i < numberOfLayers; ++i) {
    std::cout << "Layer " << i << ": from " << static_cast<double>(indexes[i]) * dx << " m to " << (static_cast<double>(indexes[i + 1]) - 1) * dx << " m" << std::endl;
  }
  std::cout << numberOfLayers << " layers in total." << std::endl;

}

void Adsorption::initialize()
{
  size_t layerCounter = 0;

  std::fill(P.begin(), P.end(), 0.0);
  std::fill(Q.begin(), Q.end(), 0.0);
  std::fill(Tgs.begin(), Tgs.end(), T);  // Solid and gas initial tempreture
  std::fill(Tw.begin(), Tw.end(), Tamb);   // Wall initial tempreture

  for (size_t j = 0; j < Ncomp; ++j) {
    components[j].Kl_values.resize(Ngrid + 1, 0.0);
    
    // ДОБАВИТЬ ЭТОТ БЛОК: Защита от пустых массивов для инертного газа
    if (components[j].isCarrierGas) {
        if (components[j].D_values.empty()) {
            components[j].D_values.resize(numberOfLayers, 0.0);
        }
        if (components[j].dH_values.empty()) {
            components[j].dH_values.resize(numberOfLayers, 0.0);
        }
    }
  }

  Kl_update(Tgs); // Grid-wise update of the mass transfer coefficients based on the initial temperature distribution

  for (size_t i = 0; i < Ngrid + 1; ++i)
  {
    if (i == indexes[layerCounter + 1]) {
      layerCounter += 1;
    }
    for (size_t j = 0; j < Ncomp; ++j)
    {
      prefactors[i * Ncomp + j] = R * ((1.0 - eps[layerCounter]) / eps[layerCounter]) * ( rho_particle[layerCounter] ) * (components[j].Kl_values[i]);
    }
  }

    // initial pressure along the column
  std::vector<double> pt_init(Ngrid + 1);

  if (!cycle) {
    // set P and Q to zero
    std::fill(P.begin(), P.end(), 0.0);
    std::fill(Q.begin(), Q.end(), 0.0);
    std::fill(Tgs.begin(), Tgs.end(), T);  // Solid and gas initial tempreture
    std::fill(Tw.begin(), Tw.end(), Tamb);   // Wall initial tempreture
    //std::fill(Tinit.begin(), Tinit.end(), Tgs[0]);   // Wall initial tempreture
    // set the initial total pressure along the column assuming the pressure gradient is constant

    std::string fileName = "C:\\InstituteWork\\ruptura_modification_examples\\Cycles_tests\\BlowDown\\testAds.txt";

    initializeFromFile(fileName, Ncomp, P, Q, Tgs, Tw);
  }
  

   for (size_t i = 0; i < Ngrid + 1; ++i) {
    Pt[i] = 0;
    for (size_t j = 0; j < Ncomp; ++j) {
      Pt[i] += P[i * Ncomp + j];
      //std::cout << Pt[i] << " " << i << std::endl;
    }
  }

  // initialize the interstitial gas velocity in the column
  for (size_t i = 0; i < Ngrid + 1; ++i)
  {
    V[i] = v_in * p_total / Pt[i];
  }

  // set the partial pressure of the carrier gas to the total initial pressure
  // for the column except for the entrance (i=0)
 // if (CarrierGasExistance) {
 // for (size_t i = 1; i < Ngrid + 1; ++i)
///  {
 //   P[i * Ncomp + carrierGasComponent] = pt_init[i];
 // }
 // }

//  else {
//    for (size_t i = 1; i < Ngrid + 1; ++i)
//  {
 //   P[i * Ncomp + 0] = pt_init[i];       // P total initialize partial pressure of the first gas component in the column
 // }
 // }


  // at the column entrance, the mol-fractions of the components in the gas phase are fixed
  // the partial pressures of the components at the entrance are the mol-fractions times the
  // total pressure
  for (size_t j = 0; j < Ncomp; ++j)
  {
    P[0 * Ncomp + j] = p_total * components[j].Yi0;
  }

  // at the entrance: mol-fractions Yi are the gas-phase mol-fractions
  // for the column: the initial mol-fraction of the carrier-gas is 1, and 0 for the other components
  //
  // the K of the carrier gas is chosen as zero
  // so Qeq is zero for all components in the column after the entrance
  // only the values for Yi at the entrance are effected by adsorption
  size_t LayerCounterSite = 0;
  //std::vector<double> QeqTmp((Ngrid + 1) * Ncomp);
  for(size_t k = 1; k < indexes.size(); ++k) {
    for(size_t i = indexes[k-1]; i < indexes[k]; ++i) {
      double sum = 0.0;
      for (size_t j = 0; j < Ncomp; ++j)
      {
        Yi[j] = std::max(P[i * Ncomp + j] / Pt[i], 0.0);
        sum += Yi[j];
      }
      for (size_t j = 0; j < Ncomp; ++j)
      {
        Yi[j] /= sum;
      }

      iastPerformance += mixture.predictMixture(LayerCounterSite, Yi, Pt[i], Xi, Ni, &cachedP0[i * Ncomp * maxIsothermTerms],
                                                &cachedPsi[i * maxIsothermTerms], Tgs[i]);

      for (size_t j = 0; j < Ncomp; ++j)
      {
        Qeq[i * Ncomp + j] = Ni[j];
      }
    }
    LayerCounterSite += 1;
  }

  for (size_t i = 0; i < Ngrid + 1; ++i)
  {
    Pt[i] = 0.0;
    for (size_t j = 0; j < Ncomp; ++j)
    {
      Pt[i] += std::max(0.0, P[i * Ncomp + j]);
    }
  }
}

void Adsorption::run()
{

 // create the output files
  std::vector<std::ofstream> streams;
  for (size_t i = 0; i < Ncomp; i++)
  {
    std::string fileName = "component_" + std::to_string(i) + "_" + components[i].name + ".data";
    streams.emplace_back(std::ofstream{fileName});
  }

  std::ofstream movieStream("column.data");

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

  for (size_t j = 0; j < Ncomp; ++j)
  {
    movieStream << "# column " << column_nr++ << ": component " << j << " Kl " << std::endl;
  }

  for (size_t step = 0; (step < Nsteps || autoSteps); ++step)
  {
    // compute new step
    computeStep(step);

    double t = static_cast<double>(step) * dt;

    if (step % writeEvery == 0)
    {
      // write breakthrough output to files
      // column 1: dimensionless time
      // column 2: time [minutes]
      // column 3: normalized partial pressure
      for (size_t j = 0; j < Ncomp; ++j)
      {
        streams[j] << t * v_in / L << " " << t / 60.0 << " "
                   << P[Ngrid * Ncomp + j] / ((p_total + dptdx * L) * components[j].Yi0) << std::endl;
      }

      for (size_t i = 0; i < Ngrid + 1; ++i)
      {
        movieStream << static_cast<double>(i) * dx << " ";
        movieStream << V[i] << " ";
        movieStream << Pt[i] << " ";
        for (size_t j = 0; j < Ncomp; ++j)
        {
          movieStream << Q[i * Ncomp + j] << " " << Qeq[i * Ncomp + j] << " " << P[i * Ncomp + j] << " "
                      << P[i * Ncomp + j] / (Pt[i] * components[j].Yi0) << " " << Dpdt[i * Ncomp + j] << " "
                      << Dqdt[i * Ncomp + j] << " ";
        }
        movieStream << Tgs[i] << " " << Tw[i] << " " << DTdt[i] << " " << DTdtWall[i] << " ";

         for (size_t j = 0; j < Ncomp; ++j)
        {
          movieStream << components[j].Kl_values[i] << " ";
        }
        
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

#ifdef PYBUILD

py::array_t<double> Breakthrough::compute()
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

void Breakthrough::setComponentsParameters(std::vector<double> molfracs, std::vector<double> params)
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

void Adsorption::computeStep(size_t step)
{
  double t = static_cast<double>(step) * dt;

  // check if we can set the expected end-time based on 10% longer time than when all
  // adorbed mol-fractions are smaller than 1% of unity
 if (autoSteps)
 {
 //   double tolerance = 0.0;
  //  for (size_t j = 0; j < Ncomp; ++j)
  //  {
  //    tolerance =
  //        std::max(tolerance, std::abs((P[Ngrid * Ncomp + j] / ((p_total + dptdx * L) * components[j].Yi0)) - 1.0));
    //}

    // consider 1% as being visibily indistinguishable from 'converged'
    // use a 10% longer time for display purposes
   // if (tolerance < 0.01)
  //  {
  //    std::cout << "\nConvergence criteria reached, running 10% longer\n\n" << std::endl;
  //    Nsteps = static_cast<size_t>(1.1 * static_cast<double>(step));
  //    autoSteps = false;
 //   }
    if (t > timeStage)
    {
      std::cout << "\nTime stage is over!\n\n" << std::endl;
      Nsteps = static_cast<size_t>(1.0 * static_cast<double>(step));
      autoSteps = false;
    }
  }

  computeCpgMix(P);

  // SSP-RK Step 1
  // ======================================================================

  // calculate the derivatives Dq/dt and Dp/dt based on Qeq, Q, V, and P
  //computeFirstDerivatives(Dqdt, Dpdt, DTdt, DTdtWall, Qeq, Qeq1, Q, V, P, Tgs, Tw);  // + DTdt, Tgs

  computeFirstDerivatives2(Dqdt, Dpdt, DTdt, DTdtWall, Qeq, Q, V, P, Tgs, Tw);  // + DTdt, Tgs

  // Dqdt and Dpdt are calculated at old time step
  // make estimate for the new loadings and new gas phase partial pressures
  // first iteration is made using the Explicit Euler scheme
  for (size_t i = 0; i < Ngrid + 1; ++i)
  {
    for (size_t j = 0; j < Ncomp; ++j)
    {
      Qnew[i * Ncomp + j] = Q[i * Ncomp + j] + dt * Dqdt[i * Ncomp + j];
      Pnew[i * Ncomp + j] = P[i * Ncomp + j] + dt * Dpdt[i * Ncomp + j];     
    }
      Tgsnew[i] = Tgs[i] + dt * DTdt[i];
      Twnew[i] = Tw[i] + dt * DTdtWall[i];
  }

  
  Kl_update(Tgsnew); // Grid-wise update of the mass transfer coefficients based on the new temperature distribution


  computeCpgMix(Pnew);

  computeEquilibriumLoadings();

  //computeVelocityTempreture();

  computeVelocityTemperatureLight();

  //computeVelocity();


  // SSP-RK Step 2
  // ======================================================================

  // calculate new derivatives at new (current) timestep
  // calculate the derivatives Dq/dt and Dp/dt based on Qeq, Q, V, and P at new (current) timestep
  //computeFirstDerivatives(Dqdtnew, Dpdtnew, DTdtnew, DTdtWallnew, Qeqnew, Qeqnew1, Qnew, Vnew, Pnew, Tgsnew, Twnew); // changed
  

  computeFirstDerivatives2(Dqdtnew, Dpdtnew, DTdtnew, DTdtWallnew, Qeqnew, Qnew, Vnew, Pnew, Tgsnew, Twnew);  // + DTdt, Tgs

  for (size_t i = 0; i < Ngrid + 1; ++i)
  { 
    for (size_t j = 0; j < Ncomp; ++j)
    {
      Qnew[i * Ncomp + j] = 0.75 * Q[i * Ncomp + j] + 0.25 * Qnew[i * Ncomp + j] + 0.25 * dt * Dqdtnew[i * Ncomp + j];
      Pnew[i * Ncomp + j] = 0.75 * P[i * Ncomp + j] + 0.25 * Pnew[i * Ncomp + j] + 0.25 * dt * Dpdtnew[i * Ncomp + j];
    }
    Tgsnew[i] = 0.75 * Tgs[i] + 0.25 * Tgsnew[i] + 0.25 * dt * DTdtnew[i];
    Twnew[i] = 0.75 * Tw[i] + 0.25 * Twnew[i] + 0.25 * dt * DTdtWallnew[i];
  }

  
  Kl_update(Tgsnew); // Grid-wise update of the mass transfer coefficients based on the new temperature distribution


  computeCpgMix(Pnew);

  computeEquilibriumLoadings();

  //computeVelocityTempreture();

  computeVelocityTemperatureLight();

  //computeVelocity();

  // SSP-RK Step 3
  // ======================================================================

  // calculate new derivatives at new (current) timestep
  // calculate the derivatives Dq/dt and Dp/dt based on Qeq, Q, V, and P at new (current) timestep
  //computeFirstDerivatives(Dqdtnew, Dpdtnew, DTdtnew, DTdtWallnew, Qeqnew, Qeqnew1, Qnew, Vnew, Pnew, Tgsnew, Twnew);

  computeFirstDerivatives2(Dqdtnew, Dpdtnew, DTdtnew, DTdtWallnew, Qeqnew, Qnew, Vnew, Pnew, Tgsnew, Twnew);  // + DTdt, Tgs

  for (size_t i = 0; i < Ngrid + 1; ++i)
  {
    for (size_t j = 0; j < Ncomp; ++j)
    {
      Qnew[i * Ncomp + j] = (1.0 / 3.0) * Q[i * Ncomp + j] + (2.0 / 3.0) * Qnew[i * Ncomp + j] +
                            (2.0 / 3.0) * dt * Dqdtnew[i * Ncomp + j];
      Pnew[i * Ncomp + j] = (1.0 / 3.0) * P[i * Ncomp + j] + (2.0 / 3.0) * Pnew[i * Ncomp + j] +
                            (2.0 / 3.0) * dt * Dpdtnew[i * Ncomp + j];
    }
    Tgsnew[i] = (1.0 / 3.0) * Tgs[i] + (2.0 / 3.0) * Tgsnew[i] +
                            (2.0 / 3.0) * dt * DTdtnew[i];
    Twnew[i] = (1.0 / 3.0) * Tw[i] + (2.0 / 3.0) * Twnew[i] +
                            (2.0 / 3.0) * dt * DTdtWallnew[i];
  }

  
  Kl_update(Tgsnew); // Grid-wise update of the mass transfer coefficients based on the new temperature distribution


  computeCpgMix(Pnew);

  computeEquilibriumLoadings();

  //computeVelocityTempreture();

  computeVelocityTemperatureLight();

  //computeVelocity();

  // update to the new time step
  std::copy(Qnew.begin(), Qnew.end(), Q.begin());
  std::copy(Pnew.begin(), Pnew.end(), P.begin());
  std::copy(Qeqnew.begin(), Qeqnew.end(), Qeq.begin());
  std::copy(Qeqnew1.begin(), Qeqnew1.end(), Qeq1.begin());
  std::copy(Vnew.begin(), Vnew.end(), V.begin());
  std::copy(Tgsnew.begin(), Tgsnew.end(), Tgs.begin());
  std::copy(Twnew.begin(), Twnew.end(), Tw.begin());

  // pulse boundary condition
  if (pulse == true)
  {
    if (t > tpulse)
    {
      for (size_t j = 0; j < Ncomp; ++j)
      {
        if (j == carrierGasComponent)
        {
          P[0 * Ncomp + j] = p_total;
        }
        else
        {
          P[0 * Ncomp + j] = 0.0;
        }
      }
    }
  }
}

void Adsorption::computeEquilibriumLoadings()
{
  size_t LayerCounterSite = 0;
  //std::vector<double> QeqTmp((Ngrid + 1) * Ncomp);
  for(size_t k = 1; k < indexes.size(); ++k) {
    for(size_t i = indexes[k-1]; i < indexes[k]; ++i) {

    Pt[i] = 0.0;
    for (size_t j = 0; j < Ncomp; ++j)
    {
      Pt[i] += std::max(0.0, Pnew[i * Ncomp + j]);
    }

      double sum = 0.0;
      for (size_t j = 0; j < Ncomp; ++j)
      {
        Yi[j] = std::max(Pnew[i * Ncomp + j] / Pt[i], 0.0);
        sum += Yi[j];
      }
      for (size_t j = 0; j < Ncomp; ++j)
      {
        Yi[j] /= sum;
      }

      iastPerformance += mixture.predictMixture(LayerCounterSite, Yi, Pt[i], Xi, Ni, &cachedP0[i * Ncomp * maxIsothermTerms],
                                                &cachedPsi[i * maxIsothermTerms], Tgsnew[i]);

      for (size_t j = 0; j < Ncomp; ++j)
      {
        Qeqnew[i * Ncomp + j] = Ni[j];
      }
    }
    LayerCounterSite += 1;
  }

  for (size_t i = 0; i < Ngrid + 1; ++i)
  {
    Pt[i] = 0.0;
    for (size_t j = 0; j < Ncomp; ++j)
    {
      Pt[i] += std::max(0.0, Pnew[i * Ncomp + j]);
    }
  }


  // check the total pressure at the outlet, it should not be negative
  if (Pt[0] + dptdx * L < 0.0)
  {
    throw std::runtime_error("Error: pressure gradient is too large (negative outlet pressure)\n");
  }
}

// calculate the derivatives Dq/dt and Dp/dt along the column

void Adsorption::computeFirstDerivatives2(
    std::vector<double> &dqdt, 
    std::vector<double> &dpdt, 
    std::vector<double> &dTdt, 
    std::vector<double> &dTdtWall,
    const std::vector<double> &q_eq, const std::vector<double> &q,
    const std::vector<double> &v, const std::vector<double> &p, 
    const std::vector<double> &Tmp, std::vector<double> &TmpWall)
{
    double idx = 1.0 / dx;
    double idx2 = 1.0 / (dx * dx);

    // =========================================================
    // 1. ГРАНИЧНАЯ ТОЧКА (Вход)
    // =========================================================
   
    for (size_t j = 0; j < Ncomp; ++j) {
        dqdt[0 * Ncomp + j] = components[j].Kl_values[0] * (q_eq[0 * Ncomp + j] - q[0 * Ncomp + j]);
        dpdt[0 * Ncomp + j] = 0.0; // P_in фиксировано
        dTdt[0] = 0.0;             // T_in фиксировано
        dTdtWall[0] = 0.0;
    }
    
    // =========================================================
    // 2. ВНУТРЕННИЕ ТОЧКИ (Цикл по пространству)
    // =========================================================
    size_t layerCounter = 0;
    for (size_t i = 1; i < Ngrid; i++)
    {
      if (i == indexes[layerCounter + 1]) {
        layerCounter += 1;
      }
      double sumH = 0.0;      
      double C_adsorbed_total = 0.0; 

      // Left points //

      // LAYER 0
      for (size_t j = 0; j < Ncomp; ++j) {
        double driving_force = (q_eq[i * Ncomp + j] - q[i * Ncomp + j]);
        dqdt[i * Ncomp + j] = components[j].Kl_values[i] * driving_force;
        sumH += components[j].dH_values[layerCounter] * components[j].Kl_values[i] * driving_force;
        C_adsorbed_total += 0; 
      }


        double dVdz = (v[i] - v[i-1]) * idx; 
        double WorkExpansion = Pt[i] * dVdz * 0; 

        double Numerator = lambda_x_layer[layerCounter] * (Tmp[i + 1] - 2 * Tmp[i] + Tmp[i - 1]) * idx2 
                         - eps[layerCounter] * (Pt[i] / (R * Tmp[i])) * Cpg_mix[i] * v[i] * (Tmp[i] - Tmp[i - 1]) * idx 
                         + (1.0 - eps[layerCounter]) * rho_particle[layerCounter] * sumH 
                         - 4.0 * h_in_layer[layerCounter] * (Tmp[i] - TmpWall[i]) / D_column_inner
                         - eps[layerCounter] * WorkExpansion;

        double DenomTerm = (Pt[i] / (R * Tmp[i])) * eps[layerCounter] * Cpg_mix[i] 
                         + (1.0 - eps[layerCounter]) * rho_particle[layerCounter] * (Cps_layer[layerCounter] + C_adsorbed_total);
        
        dTdt[i] = Numerator / DenomTerm;

        dTdtWall[i] = (4 * D_column_inner * h_in_layer[layerCounter] * (Tmp[i] - TmpWall[i]) - 4 * D_column_out * h_out * (TmpWall[i] - Tamb)) / ((std::pow(D_column_out, 2) - std::pow(D_column_inner, 2)) * rho_wall * Cpw)
                     + lambda_w * (TmpWall[i + 1] - 2.0 * TmpWall[i] + TmpWall[i - 1]) * idx2 / (rho_wall * Cpw);

        for (size_t j = 0; j < Ncomp; ++j) {
            double driving_force = (q_eq[i * Ncomp + j] - q[i * Ncomp + j]);
            // Полная формула сразу:
            dpdt[i * Ncomp + j] = 
                  (v[i - 1] * p[(i - 1) * Ncomp + j] - v[i] * p[i * Ncomp + j]) * idx 
                + components[j].D_values[layerCounter] * (p[(i + 1) * Ncomp + j] - 2.0 * p[i * Ncomp + j] + p[(i - 1) * Ncomp + j]) * idx2 
                - prefactors[i * Ncomp + j] * Tmp[i] * driving_force 
                + (v[i] * p[i * Ncomp + j] / Tmp[i]) * (Tmp[i] - Tmp[i - 1]) * idx
                + (p[i * Ncomp + j] / Tmp[i]) * dTdt[i]; // <--- Добавили dTdt сразу
      }

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
             dqdt[Ngrid * Ncomp + j] = components[j].Kl_values[Ngrid] * driving_force;
             sumH += components[j].dH_values[layerCounter] * components[j].Kl_values[Ngrid] * driving_force; 
        }

        // B. Считаем dTdt
        double dVdz = (v[Ngrid] - v[Ngrid-1]) * idx; 
        double WorkExpansion = Pt[Ngrid] * dVdz * 0; 

        double Numerator = lambda_x_layer[layerCounter] * (Tmp[Ngrid - 1] - Tmp[Ngrid]) * idx2 
                         - (Pt[Ngrid] / (R * Tmp[Ngrid])) * Cpg_mix[Ngrid] * v[Ngrid] * (Tmp[Ngrid] - Tmp[Ngrid - 1]) * idx 
                         + (1 - eps[layerCounter]) * rho_particle[layerCounter] * sumH 
                         - 4 * h_in_layer[layerCounter] * (Tmp[Ngrid] - TmpWall[Ngrid]) / D_column_inner
                         - eps[layerCounter] * WorkExpansion;

        double DenomTerm = (Pt[Ngrid] / (R * Tmp[Ngrid])) * eps[layerCounter] * Cpg_mix[Ngrid] 
                         + (1 - eps[layerCounter]) * rho_particle[layerCounter] * Cps_layer[layerCounter];
        
        dTdt[Ngrid] = Numerator / DenomTerm;

        // C. Считаем dpdt СРАЗУ
        for (size_t j = 0; j < Ncomp; ++j) {
             double driving_force = (q_eq[Ngrid * Ncomp + j] - q[Ngrid * Ncomp + j]);
             
             dpdt[Ngrid * Ncomp + j] = 
                  (v[Ngrid - 1] * p[(Ngrid - 1) * Ncomp + j] - v[Ngrid] * p[Ngrid * Ncomp + j]) * idx 
                + components[j].D_values[layerCounter] * (p[(Ngrid - 1) * Ncomp + j] - p[Ngrid * Ncomp + j]) * idx2 
                - prefactors[Ngrid * Ncomp + j] * Tmp[Ngrid] * driving_force
                + (p[Ngrid * Ncomp + j] / Tmp[Ngrid]) * dTdt[Ngrid]; // <--- dTdt тут уже известно
        }

        dTdtWall[Ngrid] = (4 * D_column_inner * h_in_layer[layerCounter] * (Tmp[Ngrid] - TmpWall[Ngrid]) - 4 * D_column_out * h_out * (TmpWall[Ngrid] - Tamb)) / ((std::pow(D_column_out, 2) - std::pow(D_column_inner, 2)) * rho_wall * Cpw)
                        + lambda_w * (TmpWall[Ngrid - 1] - TmpWall[Ngrid]) * idx2 / (rho_wall * Cpw);
    }
}

void Adsorption::Kl_update(std::vector<double> &Tmpgs)
{
  size_t layerCounter = 0;
  double Tcurrent;

  for (size_t i = 0; i < Ngrid + 1; ++i)
  {
    Tcurrent = Tmpgs[i];
    if (i == indexes[layerCounter + 1]) {
      layerCounter += 1;
    }
    for (size_t j = 0; j < Ncomp; ++j)
    {
      components[j].Kl_func(Tcurrent, layerCounter, i);
    }
  }
  
}

// calculate new velocity Vnew from Qnew, Qeqnew, Pnew, Pt
void Adsorption::computeVelocity()
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
           (components[j].D1) * (Pnew[(i - 1) * Ncomp + j] - 2.0 * Pnew[i * Ncomp + j] + Pnew[(i + 1) * Ncomp + j]) * idx2;
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
           (components[j].D1 ) * (Pnew[(i - 1) * Ncomp + j] - 2.0 * Pnew[i * Ncomp + j] + Pnew[(i + 1) * Ncomp + j]) * idx2;
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

void Adsorption::computeCpgMix(std::vector<double> &Pi) {
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

std::vector<double> Adsorption::get_pressure() {
  return P;
}

std::vector<double> Adsorption::get_q() {
  return Q;
}

std::vector<double> Adsorption::get_T() {
  return Tgs;
}


void Adsorption::computeVelocityTemperatureLight()
{
  double idx2 = 1.0 / (dx * dx);

  // first grid point
  Vnew[0] = v_in;

  // middle gridpoints
  size_t layerCounter = 0;
  for (size_t i = 1; i < Ngrid; ++i)
  {
    if (i == indexes[layerCounter + 1]) {
        layerCounter += 1;
    }
    double sum = 0.0;
    for (size_t j = 0; j < Ncomp; ++j)
    {
      sum =
          sum - prefactors[i * Ncomp + j] * Tgsnew[i] * (Qeqnew[i * Ncomp + j] - Qnew[i * Ncomp + j]) +
          components[j].D_values[layerCounter] * (Pnew[(i - 1) * Ncomp + j] - 2.0 * Pnew[i * Ncomp + j] + Pnew[(i + 1) * Ncomp + j]) * idx2;
    }
    double thermal_expansion = Vnew[i-1] * (Tgsnew[i] - Tgsnew[i-1]) / Tgsnew[i];
    // explicit version
    Vnew[i] = Vnew[i - 1] + dx * (sum - Vnew[i - 1] * dptdx) / Pt[i] + thermal_expansion + dx * (DTdtnew[i] / Tgsnew[i]);
  }

  // last grid point
  double sum = 0.0;
  for (size_t j = 0; j < Ncomp; ++j)
  {
    sum = sum - prefactors[Ngrid * Ncomp + j] * Tgsnew[Ngrid] * (Qeqnew[Ngrid * Ncomp + j] - Qnew[Ngrid * Ncomp + j]) +
          components[j].D_values[layerCounter] * (Pnew[(Ngrid - 1) * Ncomp + j] - Pnew[Ngrid * Ncomp + j]) * idx2;
  }
  //double thermal_expansion = Vnew[Ngrid] * (Tgsnew[Ngrid] - Tgsnew[Ngrid-1]) / Tgsnew[Ngrid]; // must be zero T[N] = T[N-1]
  // explicit version
  Vnew[Ngrid] = Vnew[Ngrid - 1] + dx * (sum - Vnew[Ngrid - 1] * dptdx) / Pt[Ngrid];// + dx * (DTdtnew[Ngrid] / Tgsnew[Ngrid]);
}

void Adsorption::print() const { std::cout << repr(); }

std::string Adsorption::repr() const
{
  std::string s;
  s += "Column properties\n";
  s += "=======================================================\n";
  s += "Display-name:                           " + displayName + "\n";
  s += "Temperature:                           " + std::to_string(T) + " [K]\n";
  s += "NCOMP:                               " + std::to_string(Ncomp) + " [-]\n";
  s += "Column length:                         " + std::to_string(L) + " [m]\n";
  for (size_t i = 0; i < numberOfLayers; ++i)
  {
    s += "Column void-fraction for layer " + std::to_string(i) + ":                  " + std::to_string(eps[i]) + " [-]\n";
  }
  for(size_t i = 0; i < numberOfLayers; ++i)
  {
    s += "Particle density for layer " + std::to_string(i) + ":                      " + std::to_string(rho_particle[i]) + " [kg/m^3]\n";
  }
  for(size_t i = 0; i < numberOfLayers; ++i)
  {
    s += "Particle heat capacity for layer " + std::to_string(i) + ":              " + std::to_string(Cps_layer[i]) + " [J/kg/K]\n";
  }
   s += "Wall density:                          " + std::to_string(rho_wall) + " [kg/m^3]\n";
   s += "Wall heat capacity:                    " + std::to_string(Cpw) + " [J/kg/K]\n";
   s += "Wall thermal conductivity:             " + std::to_string(lambda_w) + " [W/m/K]\n";
  for (size_t i = 0; i < numberOfLayers; ++i)
  {
    s += "Axial thermal conductivity for layer " + std::to_string(i) + ":          " + std::to_string(lambda_x_layer[i]) + " [W/m/K]\n";
  }
  s += "Total pressure:                        " + std::to_string(p_total) + " [Pa]\n";
  s += "Pressure gradient:                     " + std::to_string(dptdx) + " [Pa/m]\n";
  s += "Column entrance interstitial velocity: " + std::to_string(v_in) + " [m/s]\n";
  s += "\n\n";

  s += "Breakthrough settings\n";
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
  for (size_t i = 0; i < Ncomp; ++i)
  {
    s += components[i].repr();
    s += "\n";
  }
  return s;
}

void Adsorption::createPlotScript()
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

void Adsorption::createMovieScripts()
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
  makeMovieStream << "CALL make_movie_Kl.bat %1 %2 %3 %4\n";
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
  createMovieScriptColumnKl();
}

void Adsorption::run(std::vector<std::ofstream>& extStreams, std::ofstream& extMovieStream)
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

// -crf 18: the range of the CRF scale is 0–51, where 0 is lossless, 23 is the default,
//          and 51 is worst quality possible; 18 is visually lossless or nearly so.
// -pix_fmt yuv420p: needed on apple devices
std::string movieScriptTemplateA(std::string s)
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

void Adsorption::createMovieScriptColumnV()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_V.bat");
#else
  std::ofstream makeMovieStream("make_movie_V");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateA("V");

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

void Adsorption::createMovieScriptColumnT()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_T.bat");
#else
  std::ofstream makeMovieStream("make_movie_T");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateA("T");

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

void Adsorption::createMovieScriptColumnTw()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_Tw.bat");
#else
  std::ofstream makeMovieStream("make_movie_Tw");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateA("Tw");

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

void Adsorption::createMovieScriptColumnPt()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_Pt.bat");
#else
  std::ofstream makeMovieStream("make_movie_Pt");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateA("Pt");

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

void Adsorption::createMovieScriptColumnQ()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_Q.bat");
#else
  std::ofstream makeMovieStream("make_movie_Q");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateA("Q");

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

void Adsorption::createMovieScriptColumnQeq()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_Qeq.bat");
#else
  std::ofstream makeMovieStream("make_movie_Qeq");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateA("Qeq");

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

void Adsorption::createMovieScriptColumnKl()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_Kl.bat");
#else
  std::ofstream makeMovieStream("make_movie_Kl");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateA("Kl");

#if (__cplusplus >= 201703L)
  std::filesystem::path path{"make_movie_Kl"};
  std::filesystem::permissions(path, std::filesystem::perms::owner_exec, std::filesystem::perm_options::add);
#else
  chmod("make_movie_Kl", S_IRWXU);
#endif

  std::ofstream stream("plot_column_Kl");

  stream << "set encoding utf8\n";
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  stream << "set terminal pngcairo size ARG2,ARG3 enhanced font 'Arial,10'\n";
  stream << "set xlabel 'Adsorber position / [m]' font 'Arial,14'\n";
  stream << "set ylabel 'Mass transfer coeff., {/Arial-Italic Kl} / [1/s]' offset 0.0,0 font 'Arial,14'\n";
  stream << "set key outside top center horizontal samplen 2.5 height 0.5 spacing 1.5 font 'Arial, 10'\n";
#else
  stream << "set terminal pngcairo size ARG2,ARG3 enhanced font 'Helvetica,10'\n";
  stream << "set xlabel 'Adsorber position / [m]' font 'Helvetica,18'\n";
  stream << "set ylabel 'Mass transfer coeff., {/Helvetica-Italic Kl} / [1/s]' offset 0.0,0 font 'Helvetica,18'\n";
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
  
  // ИСПРАВЛЕНО: Поиск максимума по правильным колонкам Kl
  for (size_t i = 0; i < Ncomp; i++) {
    int col = 8 + static_cast<int>(Ncomp * 6) + static_cast<int>(i); 
    stream << "  stats 'column.data' us " << col << " nooutput\n";
    stream << "  if (max < STATS_max) { max = STATS_max; }\n";
  }
  
  stream << "stats 'column.data' us 1 nooutput\n";
  stream << "set xrange[0:STATS_max]\n";
  
  // Добавляем защиту на случай, если max = 0 (чтобы график не сломался)
  stream << "if (max == 0) { max = 1.0; }\n";
  stream << "set yrange[0:1.1*max]\n";
  
  stream << "ev=int(ARG1)\n";
  stream << "do for [i=0:int((STATS_blocks-2)/ev)] {\n";
  stream << "  plot \\\n";
  
  // ИСПРАВЛЕНО: Отрисовка линий из правильных колонок Kl
  for (size_t i = 0; i < Ncomp; i++)
  {
    int col = 8 + static_cast<int>(Ncomp * 6) + static_cast<int>(i);
    stream << "    " << "'column.data'" << " us 1:" << col << " index ev*i notitle "
           << " with li lt " << i + 1 << ",\\\n";
  }
  
  // ИСПРАВЛЕНО: Отрисовка точек из правильных колонок Kl
  for (size_t i = 0; i < Ncomp; i++)
  {
    int col = 8 + static_cast<int>(Ncomp * 6) + static_cast<int>(i);
    stream << "    " << "'column.data'" << " us 1:" << col << " index ev*i title '"
           << components[i].name << "'"
           << " with po lt " << i + 1 << (i < Ncomp - 1 ? ",\\" : "") << "\n";
  }
  stream << "}\n";
}

void Adsorption::createMovieScriptColumnP()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_P.bat");
#else
  std::ofstream makeMovieStream("make_movie_P");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateA("P");

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

void Adsorption::createMovieScriptColumnPnormalized()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_Pnorm.bat");
#else
  std::ofstream makeMovieStream("make_movie_Pnorm");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateA("Pnorm");

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

void Adsorption::createMovieScriptColumnDpdt()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_Dpdt.bat");
#else
  std::ofstream makeMovieStream("make_movie_Dpdt");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateA("Dpdt");

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

void Adsorption::createMovieScriptColumnDqdt()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_Dqdt.bat");
#else
  std::ofstream makeMovieStream("make_movie_Dqdt");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateA("Dqdt");

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