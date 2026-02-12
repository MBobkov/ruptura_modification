#pragma once

#include <string>
#include <vector>

#include "component.h"

extern bool startsWith(const std::string &str, const std::string &prefix);
extern std::string trim(const std::string &s);

/**
 * \brief Parses input files and stores simulation parameters.
 *
 * The InputReader struct is responsible for reading and parsing input files containing simulation parameters.
 * It stores all the necessary data required to set up and run simulations, including components, simulation types,
 * and various parameters related to the simulation environment.
 */
struct InputReader
{
  /**
   * \brief Constructs an InputReader and parses the given input file.
   *
   * \param fileName The name of the input file to parse.
   */
  InputReader(const std::string fileName);

  /**
   * \brief Enumerates the types of simulations supported.
   */
  enum class SimulationType
  {
    Breakthrough = 0,       ///< Breakthrough simulation.
    MixturePrediction = 1,  ///< Mixture prediction simulation.
    Fitting = 2,            ///< Fitting simulation.
    Test = 3,                ///< Test simulation.
    Pressurization = 4
  };

  std::vector<Component> components;  ///< The list of components involved in the simulation.
  size_t numberOfCarrierGases{0};     ///< The number of carrier gas components.
  size_t carrierGasComponent{0};      ///< The index of the carrier gas component.
  size_t maxIsothermTerms{0};         ///< The maximum number of isotherm terms among all components.
  size_t maxIsothermTerms1{0};         ///< The maximum number of isotherm terms among all components.
  size_t numberOfLayers{0};           ///< The layers counter.

  SimulationType simulationType{SimulationType::Breakthrough};  ///< The type of simulation to perform.
  size_t mixturePredictionMethod{0};                            ///< The method used for mixture prediction.
  size_t IASTMethod{0};                                         ///< The method used for IAST calculations.
  std::string displayName{"Column"};                            ///< The display name for the simulation.
  double temperature{433.0};                                    ///< The simulation temperature in Kelvin.
  double columnVoidFraction{0.4};                               ///< The void fraction of the column.
  double columnVoidFraction_1{0.4};                               ///< The void fraction of the column.
  double RampTime{1};                               ///< The void fraction of the column.
  double Pmin{1e5};                               ///< The void fraction of the column.

  double Cps{1000.0};                              ///< Specific heat capacity of the adsorbent (solid phase) [J/(mol·K)]
  double Cps_1{1000.0};                              ///< Specific heat capacity of the adsorbent (solid phase) [J/(mol·K)]
  double Cpw{1000.0};                              ///< Specific heat capacity of the wall material [J/(kg·K)]
  double rho_w{1000.0};                              ///< Density of the column wall [kg/m3]
 
  double lambda_ax{1000.0};                              ///< Effective axial thermal conductivity of the layer [W/(m·K)]
  double lambda_ax_1{1000.0};                              ///< Effective axial thermal conductivity of the layer [W/(m·K)]
  double lambda_w{1000.0};                              ///< Thermal conductivity of the wall material [W/(m·K)] (optional, if we consider the longitudinal conductivity of the wall)
  double h_in{1000.0};                              ///< internal heat transfer coefficient (layer→wall) [W/(m²·K)]
  double h_in_1{1000.0};                              ///< internal heat transfer coefficient (layer→wall) [W/(m²·K)]
  double h_out{1000.0};                              ///< Heat transfer coeff outside [W/(m²·K)]
  double D_column_inner{1000.0};                              ///< Diameter of inner column [m]
  double D_column_out{1000.0};                              ///< Outer Diameter of the column [m]
  double Tamb{293};
  double particleDensity{1000.0};                               ///< The density of the particles in kg/m^3.
  double particleDensity1{1000.0};                              ///< The density of the 2nd particles in kg/m^3.    
  double totalPressure{1.0e6};                                  ///< The total pressure in the system in Pa.
  double pressureGradient{0.0};                                 ///< The pressure gradient in the column.
  double columnEntranceVelocity{0.1};                           ///< The entrance velocity of the column in m/s.
  double columnLength{0.3};                                     ///< The length of the column in meters.
  double boundary_coord{0.15};                                  ///< The x-coord of the boundary
  bool CarrierGasExistance{false};
  bool IsothermalRegime{false};
  
  //double h_out_1{1000.0};                              ///< internal heat transfer coefficient (layer→wall) [W/(m²·K)]

  size_t numberOfTimeSteps{0};       ///< The number of time steps in the simulation.
  bool autoNumberOfTimeSteps{true};  ///< Whether to automatically determine the number of time steps.
  double timeStep{0.0005};           ///< The time step size in seconds.
  bool pulseBreakthrough{false};     ///< Whether to use pulse breakthrough mode.
  double pulseTime{0.0};             ///< The duration of the pulse in seconds.
  size_t printEvery{10000};          ///< The interval at which to print output.
  size_t writeEvery{10000};          ///< The interval at which to write output.
  size_t numberOfGridPoints{100};    ///< The number of grid points in the column.

  double pressureStart{-1.0};          ///< The starting pressure for isotherm calculations.
  double pressureEnd{-1.0};            ///< The ending pressure for isotherm calculations.
  size_t numberOfPressurePoints{100};  ///< The number of pressure points to calculate.
  size_t pressureScale{0};             ///< The scale for pressure calculations (0 for log, 1 for linear).

  size_t columnPressure{0};  ///< The index of the column for pressure data.
  size_t columnLoading{1};   ///< The index of the column for loading data.
  size_t columnError{2};     ///< The index of the column for error data.
};
