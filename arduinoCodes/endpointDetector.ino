
#include <Servo.h>

Servo s1,s2; 


int pos = 0;   

void setup() {
  s1.attach(9);  
  s2.attach(10);  
}

void loop() {
      s1.write(10);
      delay(500);
      s2.write(10);
      delay(500);

      //delay(5000);
      
      s1.write(10);
      delay(500);
      s2.write(100);
      delay(500);

      //delay(5000);

      s1.write(100);
      delay(500);
      s2.write(100);
      delay(500);

      //delay(5000);

      s1.write(100);
      delay(500);
      s2.write(10);
      delay(500);

      delay(5000);


}
