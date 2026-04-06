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

#include "purge.h"
#include "adsorption.h"
#include "blowdown.h"
#include "pressurization.h"
#include "cycle.h"
#include "DataReader.h"

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

Cycle::Cycle(const InputReader &inputReader): 
  Ngrid(inputReader.numberOfGridPoints),
  L(inputReader.columnLength),
  dx(L / static_cast<double>(Ngrid)),
  dt(inputReader.timeStep),
  Nsteps(inputReader.numberOfTimeSteps),
  autoSteps(inputReader.autoNumberOfTimeSteps),
  displayName(inputReader.displayName),
  components(inputReader.components),
  carrierGasComponent(inputReader.carrierGasComponent),
  Ncomp(components.size()),
  printEvery(inputReader.printEvery),
  writeEvery(inputReader.writeEvery),
  cPressurization(inputReader),
  cAdsorption(inputReader),
  cBlowdown(inputReader),
  cPurge(inputReader),
  V(Ngrid + 1),
  Pt(Ngrid + 1),
  P((Ngrid + 1) * Ncomp),
  Q((Ngrid + 1) * Ncomp),
  Qeq((Ngrid + 1) * Ncomp),
  Tgs((Ngrid + 1) * Ncomp),
  Tw((Ngrid + 1) * Ncomp),
  Dpdt((Ngrid + 1) * Ncomp),
  Dqdt((Ngrid + 1) * Ncomp),
  DTdt(Ngrid + 1),
  DTdtWall(Ngrid + 1),
  Bd(inputReader.BoundaryCoords)
  {

  }

  
  

void Cycle::initialize()
{
  
}

void Cycle::run()
{
  // Создание файлов и запись заголовков (как у вас)
  std::vector<std::ofstream> streams;
  for (size_t i = 0; i < Ncomp; i++) {
      std::string fileName = "component_" + std::to_string(i) + "_" + components[i].name + ".data";
      streams.emplace_back(fileName);
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

  // Запуск этапов цикла
  cPressurization.initialize();

  cPressurization.run(streams, movieStream);

  P = cPressurization.get_pressure();
  Q = cPressurization.get_q();
  Tgs = cPressurization.get_T();

  std::cout << "\n" << std::endl;

  std::cout << "Pressurization completed. Starting adsorption..." << std::endl;

  std::fill(Tgs.begin(), Tgs.end(), 295);
  cAdsorption.DataReciever(P, Q, Tgs, Tgs);
  cAdsorption.initialize();
  cAdsorption.run(streams, movieStream);

  P = cAdsorption.get_pressure();
  Q = cAdsorption.get_q();
  Tgs = cAdsorption.get_T();
  std::fill(Tgs.begin(), Tgs.end(), 295);

  std::cout << "\n" << std::endl;

  std::cout << "Adsorption completed. Starting blowdown..." << std::endl;

  cBlowdown.DataReciever(P, Q, Tgs, Tgs);

  cBlowdown.initialize();

  cBlowdown.run(streams, movieStream);

  P = cBlowdown.get_pressure();
  Q = cBlowdown.get_q();
  Tgs = cBlowdown.get_T();
  std::fill(Tgs.begin(), Tgs.end(), 295);

  std::cout << "\n" << std::endl;

  std::cout << "Blowdown completed. Starting purge..." << std::endl;

  reverseGridData(P, Ncomp, Ngrid);
  reverseGridData(Q, Ncomp, Ngrid);
  //std::reverse(Tgs.begin(), Tgs.end());

  cPurge.DataReciever(P, Q, Tgs, Tgs);

  cPurge.initialize();

  cPurge.run(streams, movieStream);
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

void Cycle::computeStep()
{
  //double t = static_cast<double>(step) * dt;

  cPressurization.initialize();
  std::cout << "Init" << std::endl;
  cPressurization.run();
  //std::cout << "Run" << std::endl;
  P = cPressurization.get_pressure();
  Q = cPressurization.get_q();
  Tgs = cPressurization.get_T();

}

void Cycle::print() const { std::cout << repr(); }

std::string Cycle::repr() const
{
  std::string s;
  s += "Column properties\n";
  s += "=======================================================\n";
  s += "Display-name:                           " + displayName + "\n";
  //s += "Temperature:                           " + std::to_string(T) + " [K]\n";
  ////s += "RelRight:                              " + std::to_string(relRight) + " [K]\n";
 //s += "RelLeft:                               " + std::to_string(relLeft) + " [K]\n";
  //s += "NCOMP:                               " + std::to_string(Ncomp) + " [K]\n";
  //s += "Column length:                         " + std::to_string(L) + " [m]\n";
  //s += "Column void-fraction for left layer:                  " + std::to_string(epsilon) + " [-]\n";
  //s += "Column void-fraction for left right:                  " + std::to_string(epsilon1) + " [-]\n";
  //s += "Particle density:                      " + std::to_string(rho_p) + " [kg/m^3]\n";
  //s += "2nd Particle density:                  " + std::to_string(rho_p1) + " [kg/m^3]\n";
  //s += "X-Coord of the boundary:               " + std::to_string(boundaryCoordinate) + " [m]\n";
  //s += "Total pressure:                        " + std::to_string(p_total) + " [Pa]\n";
  //s += "Pressure gradient:                     " + std::to_string(dptdx) + " [Pa/m]\n";
  //s += "Column entrance interstitial velocity: " + std::to_string(v_in) + " [m/s]\n";
  //s += "Ramp Time for pressure: " + std::to_string(ramp_time) + " [s]\n";
  //s += "Pressure minimum: " + std::to_string(Pmin) + " [Pa]\n";
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
  //s += "maximum isotherm terms for 1st layer:        " + std::to_string(maxIsothermTerms) + "\n";
  //s += "maximum isotherm terms for 2nd layer:        " + std::to_string(maxIsothermTerms1) + "\n";
  for (size_t i = 0; i < Ncomp; ++i)
  {
    s += components[i].repr();
    s += "\n";
  }
  return s;
}

void Cycle::createPlotScript()
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

  stream << "set key title '" << displayName << " {/:Italic T}=" << Tgs[0] << " K, {/:Italic p_t}=" << Pt[0] * 1e-3
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

void Cycle::createMovieScripts()
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
std::string movieScriptTemplateC(std::string s)
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

void Cycle::createMovieScriptColumnV()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_V.bat");
#else
  std::ofstream makeMovieStream("make_movie_V");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateC("V");

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
  stream << "set key title '" << displayName << " {/:Italic T}=" << Tgs[0] << " K, {/:Italic p_t}=" << Pt[0] * 1e-3
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

void Cycle::createMovieScriptColumnT()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_T.bat");
#else
  std::ofstream makeMovieStream("make_movie_T");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateC("T");

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
  stream << "set title '" << displayName << " {/:Italic Tgs_0}=" << Tgs[0] << " K, {/:Italic p_t}=" << Pt[0] * 1e-3
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

void Cycle::createMovieScriptColumnTw()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_Tw.bat");
#else
  std::ofstream makeMovieStream("make_movie_Tw");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateC("Tw");

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
  stream << "set title '" << displayName << " {/:Italic Tw_0}=" << Tgs[0] << " K, {/:Italic p_t}=" << Pt[0] * 1e-3
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

void Cycle::createMovieScriptColumnPt()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_Pt.bat");
#else
  std::ofstream makeMovieStream("make_movie_Pt");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateC("Pt");

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
  stream << "set key title '" << displayName << " {/:Italic T}=" << Tgs[0] << " K, {/:Italic p_t}=" << Pt[0] * 1e-3
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

void Cycle::createMovieScriptColumnQ()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_Q.bat");
#else
  std::ofstream makeMovieStream("make_movie_Q");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateC("Q");

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
  stream << "set key title '" << displayName << " {/:Italic T}=" << Tgs[0] << " K, {/:Italic p_t}=" << Pt[0] * 1e-3
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

void Cycle::createMovieScriptColumnQeq()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_Qeq.bat");
#else
  std::ofstream makeMovieStream("make_movie_Qeq");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateC("Qeq");

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
  stream << "set key title '" << displayName << " {/:Italic T}=" << Tgs[0] << " K, {/:Italic p_t}=" << Pt[0] * 1e-3
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

void Cycle::createMovieScriptColumnP()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_P.bat");
#else
  std::ofstream makeMovieStream("make_movie_P");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateC("P");

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
  stream << "set key title '" << displayName << " {/:Italic T}=" << Tgs[0] << " K, {/:Italic p_t}=" << Pt[0] * 1e-3
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
      for (size_t k = 0; k < Bd.size(); ++k) {
      // dt 2 - dashtype 2 (штрихованная линия)
      // lc rgb 'black' - цвет черный
      // lw 2 - толщина линии
      // graph 0 to graph 1 - линия идет от 0% до 100% высоты графика
      stream << "set arrow " << (k + 1) << " from " << Bd[k] << ", graph 0 to " << Bd[k] 
             << ", graph 1 nohead dt 2 lc rgb 'black' lw 2\n";
  }
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

void Cycle::createMovieScriptColumnPnormalized()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_Pnorm.bat");
#else
  std::ofstream makeMovieStream("make_movie_Pnorm");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateC("Pnorm");

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
  stream << "set key title '" << displayName << " {/:Italic T}=" << Tgs[0] << " K, {/:Italic p_t}=" << Pt[0] * 1e-3
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

void Cycle::createMovieScriptColumnDpdt()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_Dpdt.bat");
#else
  std::ofstream makeMovieStream("make_movie_Dpdt");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateC("Dpdt");

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
  stream << "set key title '" << displayName << " {/:Italic T}=" << Tgs[0] << " K, {/:Italic p_t}=" << Pt[0] * 1e-3
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

void Cycle::createMovieScriptColumnDqdt()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  std::ofstream makeMovieStream("make_movie_Dqdt.bat");
#else
  std::ofstream makeMovieStream("make_movie_Dqdt");
  makeMovieStream << "#!/bin/sh\n";
  makeMovieStream << "cd -- \"$(dirname \"$0\")\"\n";
#endif
  makeMovieStream << movieScriptTemplateC("Dqdt");

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
  stream << "set key title '" << displayName << " {/:Italic T}=" << Tgs[0] << " K, {/:Italic p_t}=" << Pt[0] * 1e-3
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