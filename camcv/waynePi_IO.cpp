#include "waynePi_IO.hpp"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

const string waynePi_IO::PIN_PATH = "/sys/class/gpio/gpio";
const string waynePi_IO::EXPORT_PTH = "/sys/class/gpio/export";
const string waynePi_IO::UNEXPT_PTH = "/sys/class/gpio/unexport";
const string waynePi_IO::HIGH = "1";
const string waynePi_IO::LOW = "0";

waynePi_IO::waynePi_IO(string pin, string dir)
{
  setPin(pin);
  setDir(dir);
  exprt();
  setDirection();
}

void waynePi_IO::setPin(string pin)
{
   this->pin = pin;
}

void waynePi_IO::setDir(string dir)
{
   this->dir = dir;
}

void waynePi_IO::exprt()
{
  ofstream expIO(EXPORT_PTH.c_str());
  expIO << this->pin;
  expIO.close();
}

void waynePi_IO::setDirection()
{
  string dir_path = PIN_PATH + pin +"/direction";
  ofstream dirIO(dir_path.c_str());
  dirIO << this->dir;
  dirIO.close();
}

void waynePi_IO::unexport()
{
  ofstream unexpth(UNEXPT_PTH.c_str());
  unexpth << this->pin;
  unexpth.close();
}

void waynePi_IO::high()
{
  string value_pth = PIN_PATH + pin + "/value";
  cout << value_pth << endl;
  ofstream ongpio(value_pth.c_str());
  if(ongpio < 0) cout << "could not open file" << endl;
  cout << HIGH << endl;
  ongpio << HIGH;
  ongpio.close();
}

void waynePi_IO::low()
{
  string value_pth = PIN_PATH + pin + "/value";
  ofstream offgpio(value_pth.c_str());
  offgpio << LOW;
  offgpio.close();
}

string waynePi_IO::value()
{
  string value_pth = PIN_PATH + pin + "/value";
  ifstream getval(value_pth.c_str());
  getval >> val;
  getval.close();
  return  val;
}

