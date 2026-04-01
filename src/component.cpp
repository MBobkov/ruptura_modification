#include "component.h"

#include <iostream>
#include <string>

#include "isotherm.h"

const double R = 8.31446261815324;

Component::Component(size_t _id, std::string _name, std::vector<Isotherm> _isotherms, double _Yi0, double _MolMass, double _Kl, double _Kl1, double _Kl2,
                     double _D, double _D1, double _D2, double _dH, double _dH1, double _dH2, double _Cpg, bool _isCarrierGas)
    : id(_id), name(_name), Yi0(_Yi0), MolMass(_MolMass), Kl(_Kl), Kl1(_Kl1), Kl2(_Kl2), D(_D), D1(_D1), D2(_D2), dH(_dH), dH1(_dH1), dH2(_dH2), Cpg(_Cpg), isCarrierGas(_isCarrierGas) // k1, d1, dH, dH1 added
{
  isotherm.numberOfSites = _isotherms.size();
  for (Isotherm it : _isotherms)
  {
    isotherm.add(it);
  }
}

void Component::Kl_func(double T, size_t layer, size_t indx) {
  // Check bounds for safety
  if (indx >= Kl_values.size()) {
      throw std::runtime_error("Error: Grid index (indx) out of bounds for Kl_values");
  }
  if (layer >= Kl_values_input.size() && isCarrierGas == false && (k0_values.size() <= layer || Ea_values.size() <= layer)) {
      throw std::runtime_error("Error: Layer index out of bounds for Kl_values_input");
  }

  if (isCarrierGas) {
      Kl_values[indx] = 0.0; // Carrier gas has no mass transfer resistance
      return;
  }
  // Use Arrhenius equation if parameters are available, otherwise use input value
  if (k0_values.size() > layer && Ea_values.size() > layer && isCarrierGas == false) {
      double k0 = k0_values[layer];
      double Ea = Ea_values[layer];
      Kl_values[indx] = k0 * std::exp(-Ea / (R * T));
      //std::cout << "Calculated mass transfer coefficient for layer " << layer << " at grid index " << indx << ": " << Kl_values[indx] << " 1/s (using Arrhenius parameters)" << std::endl;
  } else {
      Kl_values[indx] = Kl_values_input[layer];
  }
}

void Component::print() const { std::cout << repr(); }

std::string Component::repr() const
{
  std::string s;
  s += "Component id: " + std::to_string(id) + " [" + name + "]:\n";
  if (isCarrierGas)
  {
    s += "    carrier-gas\n";
    s += isotherm.repr();
  }
  s += "    mol-fraction in the gas:   " + std::to_string(Yi0) + " [-]\n";
  if (k0_values.size() != 0 && Ea_values.size() != 0) {
    s += "    Mass transfer coefficient with Arrhenius temperature dependence:\n";
    for (size_t i = 0; i < k0_values.size(); ++i) {
      s += "      Site " + std::to_string(i+1) + ": k0 = " + std::to_string(k0_values[i]) + " [1/s], Ea = " + std::to_string(Ea_values[i]) + " [J/mol]\n";
    }
  }
  if (!isCarrierGas)
  { 
    if (Kl_values_input.size() > 0) {
        s += "    Mass transfer coefficients (isothermal):\n";
        for (size_t i = 0; i < Kl_values_input.size(); ++i) {
            s += "      Layer " + std::to_string(i) + ": " + std::to_string(Kl_values_input[i]) + " [1/s]\n";
        }
    }
    for (size_t i = 0; i < D_values.size(); ++i) {
      s += "    diffusion coefficient for site " + std::to_string(i+1) + ": " + std::to_string(D_values[i]) + " [m^2/s]\n";
    }
    for (size_t i = 0; i < dH_values.size(); ++i) {
      s += "    Adsorption heat for site " + std::to_string(i+1) + ": " + std::to_string(dH_values[i]) + " [J/mol]\n";
    }
    s += "    Cpg:     " + std::to_string(Cpg) + " [J/(mol*K)]\n";
    s += isotherm.repr();
  }
  return s;
}
