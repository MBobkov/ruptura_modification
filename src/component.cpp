#include "component.h"

#include <iostream>
#include <string>

#include "isotherm.h"

Component::Component(size_t _id, std::string _name, std::vector<Isotherm> _isotherms, double _Yi0, double _MolMass, double _Kl, double _Kl1,
                     double _D, double _D1, double _dH, double _dH1, double _Cpg, bool _isCarrierGas)
    : id(_id), name(_name), Yi0(_Yi0), MolMass(_MolMass), Kl(_Kl), Kl1(_Kl1), D(_D), D1(_D1), dH(_dH), dH1(_dH1), Cpg(_Cpg), isCarrierGas(_isCarrierGas) // k1, d1, dH, dH1 added
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
    s += "    mas-transfer coefficient: " + std::to_string(Kl) + " [1/s]\n";
    s += "    2nd mas-transfer coefficient: " + std::to_string(Kl1) + " [1/s]\n";
    s += "    diffusion coefficient:     " + std::to_string(D) + " [m^2/s]\n";
    s += "    2nd diffusion coefficient:     " + std::to_string(D1) + " [m^2/s]\n";
    s += "    Adsorption heat [J/mol]:     " + std::to_string(dH) + " [J/mol]\n";
    s += "    2nd Adsorption heat [J/mol]:     " + std::to_string(dH1) + " [J/mol]\n";
    s += "    Cpg:     " + std::to_string(Cpg) + " [J/(mol*K)]\n";
    s += isotherm.repr();
  }
  return s;
}
