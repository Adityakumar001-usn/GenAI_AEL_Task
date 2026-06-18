import csv

prompts = [
    # Vehicle Diagnostics
    ("Vehicle Diagnostics", "Explain DTC P0420 and its common causes in a modern internal combustion engine."),
    ("Vehicle Diagnostics", "Diagnose a vehicle that has a rough idle, poor fuel economy, and a flashing check engine light."),
    ("Vehicle Diagnostics", "What does DTC P0300 indicate, and what are the first three components to inspect?"),
    ("Vehicle Diagnostics", "Explain the diagnostic procedure for DTC U0100 lost communication with ECM/PCM."),
    ("Vehicle Diagnostics", "A vehicle exhibits a parasitic draw of 1.5 Amps. Describe the step-by-step diagnostic process."),
    ("Vehicle Diagnostics", "What is the significance of fuel trim values (LTFT and STFT) when diagnosing a lean condition?"),
    ("Vehicle Diagnostics", "Describe the symptoms and diagnosis of a failing mass airflow (MAF) sensor."),

    # Predictive Maintenance
    ("Predictive Maintenance", "Predict the likely causes of premature brake pad wear on the inner pad only."),
    ("Predictive Maintenance", "How can machine learning be applied to predict turbocharger failure based on engine telemetry?"),
    ("Predictive Maintenance", "What are the early warning signs of a failing dual-clutch transmission (DCT) system?"),
    ("Predictive Maintenance", "Develop a predictive maintenance schedule for an EV fleet focusing on battery cooling systems."),
    ("Predictive Maintenance", "Explain how oil analysis is used to predict bearing failure in heavy-duty diesel engines."),
    ("Predictive Maintenance", "What sensor data streams are most valuable for predicting air suspension compressor failure?"),
    ("Predictive Maintenance", "Describe how vibration analysis is utilized to predict drivetrain component failure."),

    # CAN Bus Analysis
    ("CAN Bus Analysis", "Interpret a CAN message anomaly where the bus load suddenly spikes to 95%."),
    ("CAN Bus Analysis", "Explain the difference between a CAN 2.0A and CAN 2.0B frame structure."),
    ("CAN Bus Analysis", "A dominant state is stuck on the CAN bus. What are the potential hardware failures causing this?"),
    ("CAN Bus Analysis", "How do you use a hardware-in-the-loop (HIL) simulator to inject faults into a vehicle's CAN network?"),
    ("CAN Bus Analysis", "Describe the process of reverse engineering undocumented CAN messages for a steering angle sensor."),
    ("CAN Bus Analysis", "What is the function of the CAN bus termination resistor, and what happens if one is missing?"),
    ("CAN Bus Analysis", "Explain the arbitration process in a CAN network when two nodes transmit simultaneously."),

    # Battery Systems
    ("Battery Systems", "Explain the mechanism of thermal runaway in Lithium-ion EV batteries and preventative measures."),
    ("Battery Systems", "Compare the advantages and disadvantages of LFP vs NMC battery chemistries in automotive applications."),
    ("Battery Systems", "What is cell balancing in a Battery Management System (BMS), and why is it critical?"),
    ("Battery Systems", "Describe the state-of-charge (SOC) and state-of-health (SOH) estimation algorithms in a BMS."),
    ("Battery Systems", "Explain the impact of fast charging (DCFC) on the cycle life of EV battery packs."),
    ("Battery Systems", "What are the safety requirements for high-voltage battery disconnect units (BDU)?"),
    ("Battery Systems", "Detail the thermal management strategies (liquid vs air cooling) for high-performance EVs."),

    # ADAS
    ("ADAS", "Describe the sensor fusion process required for a reliable lane departure warning (LDW) operation."),
    ("ADAS", "How does Adaptive Cruise Control (ACC) maintain safe following distances using radar and camera data?"),
    ("ADAS", "Explain the challenges of object detection in adverse weather conditions for autonomous emergency braking (AEB)."),
    ("ADAS", "What is the role of LiDAR in level 3 autonomous driving systems compared to standard radar?"),
    ("ADAS", "Describe the calibration procedure for a forward-facing ADAS camera after a windshield replacement."),
    ("ADAS", "How do ultrasonic sensors calculate distance for automated parking assist systems?"),
    ("ADAS", "Explain the concept of operational design domain (ODD) in the context of ADAS features."),

    # Automotive Cybersecurity
    ("Automotive Cybersecurity", "Analyze a CAN spoofing attack scenario and propose mitigation strategies using SecOC."),
    ("Automotive Cybersecurity", "Explain the vulnerabilities of passive keyless entry systems to relay attacks."),
    ("Automotive Cybersecurity", "What are the requirements for securing Over-The-Air (OTA) firmware updates in modern vehicles?"),
    ("Automotive Cybersecurity", "Describe the role of a Hardware Security Module (HSM) in an automotive electronic control unit."),
    ("Automotive Cybersecurity", "How can an attacker compromise a vehicle's infotainment system via a malicious Bluetooth payload?"),
    ("Automotive Cybersecurity", "Explain the concept of defense-in-depth applied to the in-vehicle network architecture."),
    ("Automotive Cybersecurity", "What is the ISO/SAE 21434 standard, and how does it impact automotive software development?"),

    # Vehicle Dynamics
    ("Vehicle Dynamics", "Explain the behavior of a yaw rate sensor and its importance in Electronic Stability Control (ESC)."),
    ("Vehicle Dynamics", "Describe the principles of active suspension systems and their impact on ride comfort and handling."),
    ("Vehicle Dynamics", "What is slip angle, and how does it relate to lateral tire force generation?"),
    ("Vehicle Dynamics", "Explain the concept of torque vectoring and its effect on cornering performance in EVs."),
    ("Vehicle Dynamics", "How do regenerative braking algorithms blend with friction brakes to maintain seamless deceleration?"),
    ("Vehicle Dynamics", "Describe the Ackermann steering geometry and its purpose in vehicle turning."),
    ("Vehicle Dynamics", "What is the difference between understeer and oversteer, and how does ESC correct them?"),

    # Service Documentation
    ("Service Documentation", "Generate step-by-step technician instructions for replacing a high-voltage battery contactor."),
    ("Service Documentation", "Draft a service bulletin for a software update addressing false positive AEB activations."),
    ("Service Documentation", "Write a diagnostic flowchart for troubleshooting an inoperative electric power steering (EPS) system."),
    ("Service Documentation", "Create a safety protocol checklist for technicians working on 800V EV architectures."),
    ("Service Documentation", "Develop a standard operating procedure (SOP) for bleeding a brake system with ABS integration."),
    ("Service Documentation", "Write an explanation of the calibration process for a millimeter-wave radar for a service manual."),
    ("Service Documentation", "Generate a parts list and replacement procedure for a timing chain job on a dual overhead cam engine.")
]

with open('prompt_dataset.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Category', 'Prompt'])
    writer.writerows(prompts)

print(f"Generated {len(prompts)} prompts in prompt_dataset.csv")
