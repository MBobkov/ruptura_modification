#include <exception>

#include "breakthrough.h"
#include "pressurization.h"
#include "blowdown.h"
#include "purge.h"
#include "fitting.h"
#include "inputreader.h"
#include "mixture_prediction.h"
#include "special_functions.h"

int main(void)
{
  try
  {
    InputReader reader("simulation.input");

    switch (reader.simulationType)
    {
      case InputReader::SimulationType::Breakthrough:
      default:
      {
        std::cout << "Breakthrough simulation!" << std::endl;
        Breakthrough breakthrough(reader);
        
        breakthrough.print();
        breakthrough.initialize();
        breakthrough.createPlotScript();
        breakthrough.createMovieScripts();
        breakthrough.run();
        
        break;
      }
      case InputReader::SimulationType::Pressurization:
      {
        std::cout << "Pressurization simulation!" << std::endl;
        Pressurization pressurization(reader);
        
        pressurization.print();
        pressurization.initialize();
        pressurization.createPlotScript();
        pressurization.createMovieScripts();
        pressurization.run();
        
        break;
      }
       case InputReader::SimulationType::Blowdown:
      {
        std::cout << "Blowdown simulation!" << std::endl;
        Blowdown blowdown(reader);
        
        blowdown.print();
        blowdown.initialize();
        blowdown.createPlotScript();
        blowdown.createMovieScripts();
        blowdown.run();
        
        break;
      }
       case InputReader::SimulationType::Purge:
      {
        std::cout << "Purge simulation!" << std::endl;
        Purge purge(reader);
        
        purge.print();
        purge.initialize();
        purge.createPlotScript();
        purge.createMovieScripts();
        purge.run();
        
        break;
      }
      case InputReader::SimulationType::MixturePrediction:
      {
        MixturePrediction mixture(reader);

        mixture.print();
        mixture.run();
        mixture.createPureComponentsPlotScript();
        mixture.createMixturePlotScript();
        mixture.createMixtureAdsorbedMolFractionPlotScript();
        mixture.createPlotScript();
        mixture.print();
        break;
      }
      case InputReader::SimulationType::Fitting:
      {
        Fitting fitting(reader);

        fitting.run();
        break;
      }
    }
  }
  catch (std::exception const& e)
  {
    std::cerr << e.what();
    exit(-1);
  }

  return 0;
}
