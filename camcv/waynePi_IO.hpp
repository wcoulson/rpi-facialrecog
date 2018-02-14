#ifndef WAYNEPI_IO_HPP
#define WAYNEPI_IO_HPP

#include <string>
#include <iostream>
using namespace std;

class waynePi_IO
{
   private:
     static const string PIN_PATH;
     static const string EXPORT_PTH;
     static const string UNEXPT_PTH;
     static const string HIGH;
     static const string LOW;
     string pin;
     string dir;
     string val;
     void exprt();
     void setDirection();
     void setPin(string pin);
     void setDir(string dir);

   public:
     waynePi_IO(string pin, string dir);
     void unexport();
     void high();
     void low();
     string value();
};

#endif
