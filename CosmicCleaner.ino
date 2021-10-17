
#include <Servo.h>
Servo myservo;


int pos = 0;
String command;


void setup() {
  // put your setup code here, to run once:
  myservo.attach(9);

  Serial.begin(9600); 
  delay(2000);  
 
  Serial.println("Type something!");
}

void loop() {
  // put your main code here, to run repeatedly:

 if(Serial.available()){
        command = Serial.readStringUntil('\n');
         
        if(command.equals("none")){
            myservo.write(0);
        }
        else if(command.equals("plastic")){
           myservo.write(25);
           delay(3000);
           myservo.write(0);
        }
        else if(command.equals("glass")){
           myservo.write(35);
           delay(3000);
           myservo.write(0);
        }
        else if(command.equals("metal")){
            myservo.write(45);
            delay(3000);
            myservo.write(0);
        }
        else if(command.equals("waste")){
            myservo.write(55);
            delay(3000);
            myservo.write(0);
        }
         else if(command.equals("organic")){
            myservo.write(75);
            delay(3000);
            myservo.write(0);
        }
        else{
            Serial.println("Invalid command");
            myservo.write(0);
        }
    }


}
