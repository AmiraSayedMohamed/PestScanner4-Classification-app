#include <SoftwareSerial.h>

// Define RemoteXY communication
#define REMOTEXY_MODE__SOFTSERIAL
#define REMOTEXY_SERIAL_RX 0
#define REMOTEXY_SERIAL_TX 1
#define REMOTEXY_SERIAL_SPEED 9600

#include <RemoteXY.h>

// RemoteXY configuration
#pragma pack(push, 1)
uint8_t RemoteXY_CONF[] = {
  255, 6, 0, 0, 0, 31, 0, 16, 26, 0,
  5, 3, 12, 12, 106, 106, 2, 31, 0, 4,
  129, 0, 44, 5, 30, 6, 17, 72, 111, 118,
  101, 114, 98, 111, 97, 114, 100, 32, 67, 111,
  110, 116, 114, 111, 108, 0
};
struct {
  int8_t joystick_x;
  int8_t joystick_y;
  uint8_t button_stop;
  uint8_t connect_flag;
} RemoteXY;
#pragma pack(pop)

// Motor control pins
#define THROTTLE_PIN 5
#define FORWARD_PIN 6
#define REVERSE_PIN 7
#define ENABLE_PIN 8

void setup() {
  pinMode(THROTTLE_PIN, OUTPUT);
  pinMode(FORWARD_PIN, OUTPUT);
  pinMode(REVERSE_PIN, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);
  
  digitalWrite(ENABLE_PIN, HIGH); // Enable motor controller
  RemoteXY_Init();
}

void loop() {
  RemoteXY_Handler();

  if (RemoteXY.button_stop) {
    stopMotors();
    return;
  }

  int speed = map(abs(RemoteXY.joystick_y), 0, 100, 0, 255);
  int turn = RemoteXY.joystick_x;

  if (RemoteXY.joystick_y > 10) {
    moveForward(speed);
  } else if (RemoteXY.joystick_y < -10) {
    moveBackward(speed);
  } else if (abs(turn) > 20) {
    if (turn > 0) turnRight(speed);
    else turnLeft(speed);
  } else {
    stopMotors();
  }
}

// Motor control functions
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

void turnRight(int speed) {  digitalWrite(FORWARD_PIN, HIGH);
  digitalWrite(REVERSE_PIN, LOW);
  analogWrite(THROTTLE_PIN, speed / 2);
}

void stopMotors() {
  digitalWrite(FORWARD_PIN, LOW);
  digitalWrite(REVERSE_PIN, LOW);
  analogWrite(THROTTLE_PIN, 0);
}
