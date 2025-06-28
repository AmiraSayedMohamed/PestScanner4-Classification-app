#include <SoftwareSerial.h>

// Bluetooth on pins 2 (RX) and 3 (TX)
SoftwareSerial bluetooth(2, 3); // RX, TX

// Motor control pins
#define THROTTLE_PIN 5
#define FORWARD_PIN 6
#define REVERSE_PIN 7
#define ENABLE_PIN 8

char command;

void setup() {
  pinMode(THROTTLE_PIN, OUTPUT);
  pinMode(FORWARD_PIN, OUTPUT);
  pinMode(REVERSE_PIN, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);

  digitalWrite(ENABLE_PIN, HIGH); // Enable motor controller

  bluetooth.begin(9600);
  Serial.begin(9600); // For debugging
}

void loop() {
  if (bluetooth.available()) {
    command = bluetooth.read();
    Serial.println(command); // Optional: view command on Serial Monitor

    switch (command) {
      case 'F': moveForward(200); break;  // Forward
      case 'B': moveBackward(200); break; // Backward
      case 'L': turnLeft(150); break;     // Turn Left
      case 'R': turnRight(150); break;    // Turn Right
      case 'S': stopMotors(); break;      // Stop
    }
  }
}

// === Motor Control Functions ===
void moveForward(int speed) {
  digitalWrite(FORWARD_PIN, HIGH);
  digitalWrite(REVERSE_PIN, LOW);
  analogWrite(THROTTLE_PIN, speed);
}

void moveBackward(int speed) {
  digitalWrite(FORWARD_PIN, LOW);
  digitalWrite(REVERSE_PIN, HIGH);
  analogWrite(THROTTLE_PIN, speed);
}

void turnLeft(int speed) {
  digitalWrite(FORWARD_PIN, HIGH);
  digitalWrite(REVERSE_PIN, LOW);
  analogWrite(THROTTLE_PIN, speed / 2);
}

void turnRight(int speed) {
  digitalWrite(FORWARD_PIN, HIGH);
  digitalWrite(REVERSE_PIN, LOW);
  analogWrite(THROTTLE_PIN, speed / 2);
}

void stopMotors() {
  digitalWrite(FORWARD_PIN, LOW);
  digitalWrite(REVERSE_PIN, LOW);
  analogWrite(THROTTLE_PIN, 0);
}
