#include <Servo.h>
#include <Arduino.h>


Servo servo_x,servo_y;
int aa;
int bb;
String myString;
bool converted = false;
bool move_motors = false;



//--------------
#define enA 6
#define in1 5
#define in2 4

//--------------------




void servo_x_movement()
{
    servo_x.write(aa);
    delay(500);

}

void servo_y_movement()
{
    servo_y.write(bb-6);
    delay(500);
    Serial.println("done");
    move_motors = false;
    converted = false;
    aa=0;
    bb=0;
}

void spray()
{
    //pump on
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    delay(4000);

    //pump off
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);

}








void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    Serial.begin(9600);
    servo_x.attach(9);
    servo_y.attach(10);
    
    int aa = 0;
    int bb = 0;

    pinMode(enA, OUTPUT);
    pinMode(in1, OUTPUT);
    pinMode(in2, OUTPUT);
    analogWrite(enA, 250); // Send PWM signal to L298N Enable pin max upto 255
}



    


void loop() {
    
     // Define a string
    if(Serial.available() > 0 && !converted)
     {
      
         myString = Serial.readString();
                // Serial.println(myString);
                // delay(2000);
              // Variables to keep track of which number we are parsing
              bool parsingA = false;
              bool parsingB = false;

              // Iterate through each character of the string
              for (int i = 0; i < myString.length(); i++) {      
                  char ch = myString.charAt(i);
                  if (ch == 'X') {
                      parsingA = true;
                      parsingB = false;
                      continue; // Skip to next iteration
                  }
                  else if (ch == 'Y') {
                      parsingA = false;
                      parsingB = true;
                      continue; // Skip to next iteration
                  }

                  // Accumulate digits to form the number
                  if (parsingA) {
                      aa = aa * 10 + (ch - '0');
                  } else if (parsingB) {
                      bb = bb * 10 + (ch - '0');
                  }
              }
              converted=true;
              move_motors = true;
              
              
      }
     
      if (move_motors)
      {
          servo_x_movement();
          servo_y_movement();
          //spray();

      }
  
      
}
