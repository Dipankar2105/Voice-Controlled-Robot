# HARDWARE SAFETY AND COMMISSIONING MANUAL

## 1. Electrical Invariants

### 1.1 Ultrasonic Echo Voltage Divider (CRITICAL)
- **Sensor:** HC-SR04 ultrasonic distance sensor operates at 5.0 V VCC.
- **Echo Output:** 5.0 V TTL pulse.
- **Pi GPIO Limit:** 3.3 V max. Raspberry Pi GPIO pins have no internal 5V clamp.
- **Requirement:** A passive resistive voltage divider MUST be installed between the HC-SR04 ECHO pin and Raspberry Pi GPIO 24.
  - Recommended: $R_1 = 1.0\text{ k}\Omega$ (series between ECHO and GPIO 24), $R_2 = 2.0\text{ k}\Omega$ (shunt between GPIO 24 and GND).
  - Calculated output: $V_{out} = 5.0\text{ V} \times \frac{2000}{3000} = 3.33\text{ V}$.

### 1.2 Motor Power Supply Isolation
- **Driver:** L298N Dual H-Bridge.
- **Power:** Four DC motors MUST be energized exclusively by a dedicated motor battery pack (e.g., 7.4V 2S Li-ion or 6xAA battery pack) connected to the L298N power terminal ($V_{MS}$) and GND.
- **NEVER** power motors from the Raspberry Pi 5V or 3.3V header pins.
- **Common Ground:** The negative battery lead, L298N GND, and Raspberry Pi GND must be securely connected together to establish a shared ground reference.

## 2. Sensor Coverage Invariants

### 2.1 Single Forward Sensor Limitation
- The HC-SR04 is installed on the front bumper.
- Beam angle: $15^\circ - 30^\circ$.
- Blind zone: Under $2.0\text{ cm}$, ultrasonic ring-down produces erratic or zero readings. Any reading $< 2.0\text{ cm}$ or sensor timeout must be treated as BLOCKED / FAULT.
- **Zero Lateral/Rear Sensing:** The robot has NO sensors facing left, right, or rear. The software must NEVER state that turns or reversing are verified clear. Turns must be strictly bounded in place, and reverse movements must be conservative or restricted.

## 3. Physical Bench Commissioning Checklist

Before running any test on physical hardware:
1. Elevate the robot chassis so all four wheels rotate freely in the air.
2. Confirm the voltage divider on GPIO 24 using a multimeter.
3. Verify common ground connectivity between Pi GND and L298N GND.
4. Have a physical battery cutoff switch accessible at all times.
5. Execute static configuration preflight before energizing motor power.
