#include "component.h"

#include <iostream>
#include <string>

#include "isotherm.h"

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
  if (!isCarrierGas)
  {
    for (size_t i = 0; i < Kl_values.size(); ++i) {
      s += "    mas-transfer coefficient for site " + std::to_string(i+1) + ": " + std::to_string(Kl_values[i]) + " [1/s]\n";
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
