"""
SCARA Robot Trajectory Controller for MyActuator X-V3 Motors
=============================================================
Sistema: Braccio SCARA orizzontale 2-DOF
Motori: MyActuator RMD-X8 con driver V3
Protocollo: CAN bus / RS485

Questo modulo implementa:
- Protocollo comunicazione MyActuator V3 (CAN e RS485)
- Controllore di traiettoria che invia i percorsi pianificati ai motori
- Monitoraggio stato motori e sicurezza
- Integrazione con il simulatore scara_pick_place.py

Utilizzo:
    python scara_controller.py              # Simula + controlla motori
    python scara_controller.py --dry-run    # Solo simulazione (senza hardware)

Autore: Claude
"""

import struct
import time
import threading
import argparse
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Dict

# Importa il simulatore esistente
from scara_pick_place import (
    SCARAParams, Waypoint, TrajectoryConstraints, ArmConfig, VelocityProfile,
    generate_trajectory_with_payload, simulate_trajectory, analyze_feasibility,
    plot_results
)


# =============================================================================
# COSTANTI PROTOCOLLO V3
# =============================================================================

class MotorCommand(Enum):
    """Comandi del protocollo MyActuator V3."""
    # Lettura PID
    READ_PID = 0x30
    WRITE_PID_RAM = 0x31
    WRITE_PID_ROM = 0x32

    # Accelerazione
    READ_ACCEL = 0x42
    WRITE_ACCEL = 0x43

    # Encoder
    READ_ENCODER_MULTI = 0x60
    READ_ENCODER_MULTI_RAW = 0x61
    READ_ENCODER_ZERO_OFFSET = 0x62
    WRITE_ENCODER_ZERO = 0x63
    WRITE_ENCODER_ZERO_CURRENT = 0x64

    # Encoder single-turn
    READ_ENCODER_SINGLE = 0x90
    READ_MULTI_TURN_ANGLE = 0x92
    READ_SINGLE_TURN_ANGLE = 0x94

    # Stato motore
    READ_STATUS_1 = 0x9A  # Temperatura, tensione, errori
    READ_STATUS_2 = 0x9C  # Temperatura, corrente, velocità, angolo
    READ_STATUS_3 = 0x9D  # Temperatura, correnti di fase

    # Controllo motore
    MOTOR_SHUTDOWN = 0x80
    MOTOR_STOP = 0x81

    # Controllo closed-loop
    TORQUE_CONTROL = 0xA1      # Controllo coppia/corrente
    SPEED_CONTROL = 0xA2       # Controllo velocità
    ABS_POSITION_CONTROL = 0xA4  # Controllo posizione assoluta
    SINGLE_TURN_CONTROL = 0xA6  # Controllo posizione singolo giro
    INC_POSITION_CONTROL = 0xA8  # Controllo posizione incrementale

    # Sistema
    READ_OPERATING_MODE = 0x70
    SYSTEM_RESET = 0x76
    BRAKE_RELEASE = 0x77
    BRAKE_LOCK = 0x78
    READ_RUNTIME = 0xB1
    READ_VERSION = 0xB2
    SET_COMM_TIMEOUT = 0xB3
    SET_BAUD_RATE = 0xB4
    READ_MODEL = 0xB5


class AccelIndex(Enum):
    """Indici per il comando accelerazione (0x42/0x43)."""
    POSITION_ACCEL = 0x00
    POSITION_DECEL = 0x01
    SPEED_ACCEL = 0x02
    SPEED_DECEL = 0x03


class PIDIndex(Enum):
    """Indici per i comandi PID (0x30/0x31/0x32)."""
    CURRENT_KP = 0x01
    CURRENT_KI = 0x02
    SPEED_KP = 0x04
    SPEED_KI = 0x05
    POSITION_KP = 0x07
    POSITION_KI = 0x08
    POSITION_KD = 0x09


# Errori motore (da 0x9A)
MOTOR_ERRORS = {
    0x0002: "Stallo motore",
    0x0004: "Bassa tensione",
    0x0008: "Sovratensione",
    0x0010: "Sovracorrente",
    0x0040: "Sovrapotenza",
    0x0080: "Errore scrittura parametri calibrazione",
    0x0100: "Velocità eccessiva",
    0x1000: "Sovratemperatura motore",
    0x2000: "Errore calibrazione encoder",
}


# =============================================================================
# PROTOCOLLO CAN/RS485
# =============================================================================

@dataclass
class MotorStatus:
    """Stato corrente del motore."""
    temperature: int = 0        # °C
    voltage: float = 0.0        # V
    torque_current: float = 0.0 # A
    speed: int = 0              # dps (output shaft)
    angle: int = 0              # degrees (output shaft)
    error_state: int = 0
    brake_released: bool = False
    timestamp: float = 0.0

    @property
    def errors(self) -> List[str]:
        """Decodifica flag errori."""
        errs = []
        for bit, desc in MOTOR_ERRORS.items():
            if self.error_state & bit:
                errs.append(desc)
        return errs

    @property
    def has_error(self) -> bool:
        return self.error_state != 0


def compute_crc16(data: bytes) -> int:
    """Calcola CRC16 per protocollo RS485 (Modbus CRC16)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


class BusInterface(Enum):
    """Tipo di interfaccia bus."""
    CAN = "can"
    RS485 = "rs485"


class MotorProtocol:
    """
    Implementazione del protocollo di comunicazione MyActuator V3.

    Supporta sia CAN bus che RS485.
    Il protocollo usa frame da 8 byte per i dati, identici tra CAN e RS485.
    """

    CAN_BASE_SEND = 0x140   # ID CAN invio: 0x140 + motor_id
    CAN_BASE_REPLY = 0x240  # ID CAN risposta: 0x240 + motor_id
    RS485_HEADER = 0x3E     # Header frame RS485
    RS485_DATA_LEN = 8      # Lunghezza dati fissa

    def __init__(self, bus_type: BusInterface = BusInterface.CAN,
                 interface: str = "socketcan", channel: str = "can0",
                 serial_port: str = "/dev/ttyUSB0", baudrate: int = 115200):
        """
        Inizializza il protocollo.

        Args:
            bus_type: Tipo di bus (CAN o RS485)
            interface: Interfaccia CAN (socketcan, pcan, ecc.)
            channel: Canale CAN (can0, can1, ecc.)
            serial_port: Porta seriale per RS485
            baudrate: Baud rate RS485
        """
        self.bus_type = bus_type
        self.bus = None
        self._interface = interface
        self._channel = channel
        self._serial_port = serial_port
        self._baudrate = baudrate

    def connect(self):
        """Connessione al bus."""
        if self.bus_type == BusInterface.CAN:
            try:
                import can
                self.bus = can.interface.Bus(
                    interface=self._interface,
                    channel=self._channel,
                    bitrate=1000000  # 1Mbps standard V3
                )
                print(f"   CAN bus connesso: {self._channel} @ 1Mbps")
            except ImportError:
                raise RuntimeError(
                    "Libreria python-can non trovata. "
                    "Installare con: pip install python-can"
                )
        elif self.bus_type == BusInterface.RS485:
            try:
                import serial
                self.bus = serial.Serial(
                    port=self._serial_port,
                    baudrate=self._baudrate,
                    bytesize=8,
                    stopbits=1,
                    parity='N',
                    timeout=0.1
                )
                print(f"   RS485 connesso: {self._serial_port} @ {self._baudrate}")
            except ImportError:
                raise RuntimeError(
                    "Libreria pyserial non trovata. "
                    "Installare con: pip install pyserial"
                )

    def disconnect(self):
        """Disconnessione dal bus."""
        if self.bus is not None:
            self.bus.shutdown() if self.bus_type == BusInterface.CAN else self.bus.close()
            self.bus = None
            print("   Bus disconnesso")

    def _build_data(self, command: int, data: list) -> bytes:
        """Costruisci frame dati da 8 byte."""
        frame = [command] + data
        while len(frame) < 8:
            frame.append(0x00)
        return bytes(frame[:8])

    def send_command(self, motor_id: int, command: int, data: list = None,
                     expect_reply: bool = True, timeout: float = 0.1) -> Optional[bytes]:
        """
        Invia comando al motore e attendi risposta.

        Args:
            motor_id: ID del motore (1-32)
            command: Byte comando (es. 0xA4)
            data: Lista byte dati (max 7 byte, senza il byte comando)
            expect_reply: Se attendere risposta
            timeout: Timeout risposta in secondi

        Returns:
            8 byte di risposta o None
        """
        if data is None:
            data = []

        frame_data = self._build_data(command, data)

        if self.bus_type == BusInterface.CAN:
            return self._send_can(motor_id, frame_data, expect_reply, timeout)
        else:
            return self._send_rs485(motor_id, frame_data, expect_reply, timeout)

    def _send_can(self, motor_id: int, data: bytes,
                  expect_reply: bool, timeout: float) -> Optional[bytes]:
        """Invio via CAN bus."""
        import can

        can_id = self.CAN_BASE_SEND + motor_id
        msg = can.Message(
            arbitration_id=can_id,
            data=data,
            is_extended_id=False
        )
        self.bus.send(msg)

        if not expect_reply:
            return None

        reply_id = self.CAN_BASE_REPLY + motor_id
        reply = self.bus.recv(timeout=timeout)
        if reply and reply.arbitration_id == reply_id:
            return bytes(reply.data)
        return None

    def _send_rs485(self, motor_id: int, data: bytes,
                    expect_reply: bool, timeout: float) -> Optional[bytes]:
        """Invio via RS485."""
        # Frame: header(0x3E) + ID + length(8) + data(8) + CRC16(2)
        frame = bytes([self.RS485_HEADER, motor_id, self.RS485_DATA_LEN]) + data
        crc = compute_crc16(frame)
        frame += struct.pack('<H', crc)

        self.bus.write(frame)
        self.bus.flush()

        if not expect_reply:
            return None

        # Leggi risposta: header + ID + length + data(8) + CRC16(2) = 13 byte
        self.bus.timeout = timeout
        response = self.bus.read(13)
        if len(response) == 13:
            # Verifica CRC
            crc_recv = struct.unpack('<H', response[11:13])[0]
            crc_calc = compute_crc16(response[:11])
            if crc_recv == crc_calc:
                return response[3:11]  # Restituisci solo i dati
        return None


# =============================================================================
# DRIVER MOTORE V3
# =============================================================================

class V3Motor:
    """
    Driver per singolo motore MyActuator con protocollo V3.

    Gestisce comunicazione, stato e comandi di controllo.
    """

    def __init__(self, motor_id: int, protocol: MotorProtocol,
                 gear_ratio: float = 9.0):
        """
        Args:
            motor_id: ID motore sul bus (1-32)
            protocol: Istanza protocollo comunicazione
            gear_ratio: Rapporto di riduzione
        """
        self.motor_id = motor_id
        self.protocol = protocol
        self.gear_ratio = gear_ratio
        self.status = MotorStatus()
        self._lock = threading.Lock()

    def _send(self, command: MotorCommand, data: list = None,
              expect_reply: bool = True) -> Optional[bytes]:
        """Invia comando con thread-safety."""
        with self._lock:
            return self.protocol.send_command(
                self.motor_id, command.value, data, expect_reply
            )

    # --- Comandi di stato ---

    def read_status_1(self) -> MotorStatus:
        """Leggi temperatura, tensione e errori (0x9A)."""
        reply = self._send(MotorCommand.READ_STATUS_1)
        if reply:
            self.status.temperature = struct.unpack('b', bytes([reply[1]]))[0]
            self.status.brake_released = (reply[3] == 0x01)
            self.status.voltage = struct.unpack('<H', reply[4:6])[0] * 0.1
            self.status.error_state = struct.unpack('<H', reply[6:8])[0]
            self.status.timestamp = time.time()
        return self.status

    def read_status_2(self) -> MotorStatus:
        """Leggi temperatura, corrente, velocità e angolo (0x9C)."""
        reply = self._send(MotorCommand.READ_STATUS_2)
        if reply:
            self.status.temperature = struct.unpack('b', bytes([reply[1]]))[0]
            self.status.torque_current = struct.unpack('<h', reply[2:4])[0] * 0.01
            self.status.speed = struct.unpack('<h', reply[4:6])[0]
            self.status.angle = struct.unpack('<h', reply[6:8])[0]
            self.status.timestamp = time.time()
        return self.status

    def read_multi_turn_angle(self) -> float:
        """Leggi angolo multi-giro in gradi (0x92)."""
        reply = self._send(MotorCommand.READ_MULTI_TURN_ANGLE)
        if reply:
            raw = struct.unpack('<i', reply[4:8])[0]
            return raw * 0.01  # 0.01°/LSB
        return 0.0

    # --- Comandi di controllo motore ---

    def shutdown(self):
        """Spegni il motore (0x80). Cancella tutti gli stati."""
        self._send(MotorCommand.MOTOR_SHUTDOWN)

    def stop(self):
        """Ferma il motore (0x81). Mantiene modo closed-loop."""
        self._send(MotorCommand.MOTOR_STOP)

    def brake_release(self):
        """Rilascia il freno (0x77)."""
        self._send(MotorCommand.BRAKE_RELEASE)

    def brake_lock(self):
        """Blocca il freno (0x78)."""
        self._send(MotorCommand.BRAKE_LOCK)

    def system_reset(self):
        """Reset sistema (0x76)."""
        self._send(MotorCommand.SYSTEM_RESET, expect_reply=False)

    # --- Controllo coppia ---

    def torque_control(self, current_a: float) -> Optional[bytes]:
        """
        Controllo coppia/corrente (0xA1).

        Args:
            current_a: Corrente target in Ampere

        Returns:
            Risposta dal motore
        """
        iq = int(current_a / 0.01)  # 0.01A/LSB
        iq = max(-32768, min(32767, iq))
        data = [0x00, 0x00, 0x00,
                iq & 0xFF, (iq >> 8) & 0xFF,
                0x00, 0x00]
        reply = self._send(MotorCommand.TORQUE_CONTROL, data)
        if reply:
            self._parse_control_reply(reply)
        return reply

    # --- Controllo velocità ---

    def speed_control(self, speed_dps: float) -> Optional[bytes]:
        """
        Controllo velocità (0xA2).

        Args:
            speed_dps: Velocità target in dps (output shaft), 0.01dps/LSB

        Returns:
            Risposta dal motore
        """
        speed_raw = int(speed_dps / 0.01)  # 0.01dps/LSB
        data = [0x00, 0x00, 0x00,
                speed_raw & 0xFF, (speed_raw >> 8) & 0xFF,
                (speed_raw >> 16) & 0xFF, (speed_raw >> 24) & 0xFF]
        reply = self._send(MotorCommand.SPEED_CONTROL, data)
        if reply:
            self._parse_control_reply(reply)
        return reply

    # --- Controllo posizione assoluta ---

    def absolute_position_control(self, angle_deg: float,
                                  max_speed_dps: int = 500) -> Optional[bytes]:
        """
        Controllo posizione assoluta multi-giro (0xA4).

        Args:
            angle_deg: Angolo target in gradi (output shaft)
            max_speed_dps: Velocità massima in dps (output shaft)

        Returns:
            Risposta dal motore
        """
        angle_raw = int(angle_deg / 0.01)  # 0.01°/LSB
        speed_raw = max(0, min(65535, max_speed_dps))  # uint16_t, 1dps/LSB

        data = [0x00,
                speed_raw & 0xFF, (speed_raw >> 8) & 0xFF,
                angle_raw & 0xFF, (angle_raw >> 8) & 0xFF,
                (angle_raw >> 16) & 0xFF, (angle_raw >> 24) & 0xFF]
        reply = self._send(MotorCommand.ABS_POSITION_CONTROL, data)
        if reply:
            self._parse_control_reply(reply)
        return reply

    # --- Controllo posizione incrementale ---

    def incremental_position_control(self, angle_inc_deg: float,
                                     max_speed_dps: int = 500) -> Optional[bytes]:
        """
        Controllo posizione incrementale (0xA8).

        Args:
            angle_inc_deg: Incremento angolare in gradi
            max_speed_dps: Velocità massima in dps

        Returns:
            Risposta dal motore
        """
        angle_raw = int(angle_inc_deg / 0.01)  # 0.01°/LSB
        speed_raw = max(0, min(65535, max_speed_dps))

        data = [0x00,
                speed_raw & 0xFF, (speed_raw >> 8) & 0xFF,
                angle_raw & 0xFF, (angle_raw >> 8) & 0xFF,
                (angle_raw >> 16) & 0xFF, (angle_raw >> 24) & 0xFF]
        reply = self._send(MotorCommand.INC_POSITION_CONTROL, data)
        if reply:
            self._parse_control_reply(reply)
        return reply

    # --- Configurazione ---

    def set_acceleration(self, index: AccelIndex, accel_dps2: int):
        """
        Imposta accelerazione (0x43). Range: 100-60000 dps/s.

        Args:
            index: Tipo di accelerazione
            accel_dps2: Accelerazione in dps/s
        """
        accel = max(100, min(60000, accel_dps2))
        data = [index.value, 0x00, 0x00,
                accel & 0xFF, (accel >> 8) & 0xFF,
                (accel >> 16) & 0xFF, (accel >> 24) & 0xFF]
        self._send(MotorCommand.WRITE_ACCEL, data)

    def set_pid(self, index: PIDIndex, value: float, to_rom: bool = False):
        """
        Imposta parametro PID (0x31 RAM / 0x32 ROM).

        Args:
            index: Parametro PID
            value: Valore float
            to_rom: Se salvare in ROM (persistente)
        """
        raw = struct.pack('<f', value)
        cmd = MotorCommand.WRITE_PID_ROM if to_rom else MotorCommand.WRITE_PID_RAM
        data = [index.value, 0x00, 0x00,
                raw[0], raw[1], raw[2], raw[3]]
        self._send(cmd, data)

    def set_communication_timeout(self, timeout_ms: int):
        """
        Imposta timeout protezione comunicazione (0xB3).

        Args:
            timeout_ms: Timeout in ms (0 = disabilitato)
        """
        data = [0x00, 0x00, 0x00,
                timeout_ms & 0xFF, (timeout_ms >> 8) & 0xFF,
                (timeout_ms >> 16) & 0xFF, (timeout_ms >> 24) & 0xFF]
        self._send(MotorCommand.SET_COMM_TIMEOUT, data)

    def set_encoder_zero_current(self):
        """Imposta posizione attuale come zero encoder (0x64)."""
        self._send(MotorCommand.WRITE_ENCODER_ZERO_CURRENT)

    def _parse_control_reply(self, reply: bytes):
        """Parsa risposta standard dai comandi di controllo."""
        self.status.temperature = struct.unpack('b', bytes([reply[1]]))[0]
        self.status.torque_current = struct.unpack('<h', reply[2:4])[0] * 0.01
        self.status.speed = struct.unpack('<h', reply[4:6])[0]
        self.status.angle = struct.unpack('<h', reply[6:8])[0]
        self.status.timestamp = time.time()

    # --- Conversione angoli giunto <-> motore ---

    def joint_to_motor_angle(self, joint_angle_rad: float) -> float:
        """Converti angolo giunto (rad) in angolo motore (gradi output shaft)."""
        return np.rad2deg(joint_angle_rad)

    def motor_to_joint_angle(self, motor_angle_deg: float) -> float:
        """Converti angolo motore (gradi output shaft) in angolo giunto (rad)."""
        return np.deg2rad(motor_angle_deg)


# =============================================================================
# SIMULAZIONE MOTORE (per dry-run senza hardware)
# =============================================================================

class SimulatedMotor(V3Motor):
    """
    Motore simulato per test senza hardware reale.
    Emula il comportamento del protocollo V3 in software.
    """

    def __init__(self, motor_id: int, gear_ratio: float = 9.0):
        # Crea un protocollo fittizio
        protocol = MotorProtocol(bus_type=BusInterface.CAN)
        super().__init__(motor_id, protocol, gear_ratio)
        self._current_angle_deg = 0.0
        self._target_angle_deg = 0.0
        self._max_speed_dps = 500
        self.status = MotorStatus(voltage=48.0, temperature=35)

    def _send(self, command, data=None, expect_reply=True):
        """Override: simula risposta."""
        return bytes(8)  # Risposta fittizia

    def read_status_1(self) -> MotorStatus:
        self.status.timestamp = time.time()
        return self.status

    def read_status_2(self) -> MotorStatus:
        self.status.timestamp = time.time()
        return self.status

    def read_multi_turn_angle(self) -> float:
        return self._current_angle_deg

    def absolute_position_control(self, angle_deg, max_speed_dps=500):
        self._target_angle_deg = angle_deg
        self._max_speed_dps = max_speed_dps
        # Simula movimento istantaneo (nel controller reale il motore si muove)
        self._current_angle_deg = angle_deg
        self.status.angle = int(angle_deg) % 32768
        self.status.timestamp = time.time()
        return bytes(8)

    def incremental_position_control(self, angle_inc_deg, max_speed_dps=500):
        self._target_angle_deg = self._current_angle_deg + angle_inc_deg
        self._current_angle_deg = self._target_angle_deg
        self.status.angle = int(self._current_angle_deg) % 32768
        self.status.timestamp = time.time()
        return bytes(8)

    def shutdown(self): pass
    def stop(self): pass
    def brake_release(self): pass
    def brake_lock(self): pass
    def system_reset(self): pass
    def set_acceleration(self, index, accel): pass
    def set_communication_timeout(self, timeout_ms): pass
    def set_encoder_zero_current(self): pass


# =============================================================================
# CONTROLLORE TRAIETTORIA
# =============================================================================

@dataclass
class ControllerConfig:
    """Configurazione del controllore traiettoria."""
    # Frequenza di controllo
    control_rate_hz: float = 100.0     # Hz (intervallo invio comandi)

    # Sicurezza
    max_position_error_deg: float = 5.0   # Errore posizione massimo ammesso
    comm_timeout_ms: int = 500            # Timeout comunicazione motore
    max_motor_temp: int = 80              # Temperatura massima motore (°C)

    # Velocità massima per i comandi di posizione
    max_motor_speed_dps: int = 720   # dps output shaft

    # Accelerazione motore (per planning interno motore)
    position_accel_dps2: int = 10000  # dps/s
    position_decel_dps2: int = 10000  # dps/s

    # Abilitazioni
    enable_safety_checks: bool = True
    enable_status_monitoring: bool = True


class TrajectoryController:
    """
    Controllore di traiettoria per SCARA 2-DOF con motori X-V3.

    Prende la traiettoria pianificata dal simulatore (angoli giunto nel tempo)
    e invia i comandi di posizione ai motori alla frequenza di controllo.

    Modalità di controllo:
    1. Posizione assoluta (0xA4): invia posizioni target lungo la traiettoria
    2. Il motore V3 gestisce internamente il planning della velocità
    """

    def __init__(self, motor1: V3Motor, motor2: V3Motor,
                 config: ControllerConfig = None):
        """
        Args:
            motor1: Driver motore giunto 1 (spalla)
            motor2: Driver motore giunto 2 (gomito)
            config: Configurazione controllore
        """
        self.motor1 = motor1
        self.motor2 = motor2
        self.config = config or ControllerConfig()

        self._running = False
        self._emergency_stop = False
        self._control_thread = None

        # Log esecuzione
        self.log_time = []
        self.log_q1_cmd = []
        self.log_q2_cmd = []
        self.log_q1_actual = []
        self.log_q2_actual = []
        self.log_tau1 = []
        self.log_tau2 = []
        self.log_temp1 = []
        self.log_temp2 = []

    def initialize(self):
        """Inizializza motori per il controllo traiettoria."""
        print("\n   Inizializzazione motori...")

        # 1. Rilascio freni
        self.motor1.brake_release()
        self.motor2.brake_release()
        time.sleep(0.1)

        # 2. Imposta timeout comunicazione
        self.motor1.set_communication_timeout(self.config.comm_timeout_ms)
        self.motor2.set_communication_timeout(self.config.comm_timeout_ms)

        # 3. Configura accelerazioni per il position planning del motore
        for motor in [self.motor1, self.motor2]:
            motor.set_acceleration(
                AccelIndex.POSITION_ACCEL,
                self.config.position_accel_dps2
            )
            motor.set_acceleration(
                AccelIndex.POSITION_DECEL,
                self.config.position_decel_dps2
            )

        # 4. Verifica stato motori
        status1 = self.motor1.read_status_1()
        status2 = self.motor2.read_status_1()

        if status1.has_error:
            print(f"   ERRORE Motore 1: {', '.join(status1.errors)}")
            return False
        if status2.has_error:
            print(f"   ERRORE Motore 2: {', '.join(status2.errors)}")
            return False

        print(f"   Motore 1: T={status1.temperature}°C, V={status1.voltage:.1f}V")
        print(f"   Motore 2: T={status2.temperature}°C, V={status2.voltage:.1f}V")
        print("   Inizializzazione completata")
        return True

    def emergency_stop(self):
        """Arresto di emergenza."""
        self._emergency_stop = True
        self._running = False
        self.motor1.stop()
        self.motor2.stop()
        time.sleep(0.05)
        self.motor1.brake_lock()
        self.motor2.brake_lock()
        print("\n   ARRESTO DI EMERGENZA!")

    def shutdown(self):
        """Spegni motori in sicurezza."""
        self._running = False
        if self._control_thread and self._control_thread.is_alive():
            self._control_thread.join(timeout=2.0)

        self.motor1.stop()
        self.motor2.stop()
        time.sleep(0.05)
        self.motor1.brake_lock()
        self.motor2.brake_lock()
        self.motor1.shutdown()
        self.motor2.shutdown()
        print("   Motori spenti")

    def execute_trajectory(self, t: np.ndarray, q: np.ndarray,
                          qd: np.ndarray, cart: dict,
                          blocking: bool = True) -> bool:
        """
        Esegui la traiettoria pianificata sui motori.

        Args:
            t: Array tempi [s]
            q: Array angoli giunti [2, N] in radianti
            qd: Array velocità giunti [2, N] in rad/s
            cart: Dizionario dati cartesiani (dal simulatore)
            blocking: Se attendere completamento

        Returns:
            True se completata con successo
        """
        if self._emergency_stop:
            print("   Sistema in stato di emergenza. Reset necessario.")
            return False

        # Sottocampiona la traiettoria alla frequenza di controllo
        dt_ctrl = 1.0 / self.config.control_rate_hz
        t_ctrl = np.arange(t[0], t[-1], dt_ctrl)

        # Interpola angoli alla frequenza di controllo
        from scipy.interpolate import interp1d
        q1_interp = interp1d(t, q[0], kind='linear', fill_value='extrapolate')
        q2_interp = interp1d(t, q[1], kind='linear', fill_value='extrapolate')
        qd1_interp = interp1d(t, qd[0], kind='linear', fill_value='extrapolate')
        qd2_interp = interp1d(t, qd[1], kind='linear', fill_value='extrapolate')

        q1_ctrl = q1_interp(t_ctrl)
        q2_ctrl = q2_interp(t_ctrl)
        qd1_ctrl = qd1_interp(t_ctrl)
        qd2_ctrl = qd2_interp(t_ctrl)

        n_points = len(t_ctrl)

        print(f"\n   Esecuzione traiettoria:")
        print(f"   Punti di controllo: {n_points}")
        print(f"   Frequenza: {self.config.control_rate_hz} Hz")
        print(f"   Durata: {t_ctrl[-1]:.2f} s")

        # Resetta log
        self.log_time = []
        self.log_q1_cmd = []
        self.log_q2_cmd = []
        self.log_q1_actual = []
        self.log_q2_actual = []
        self.log_tau1 = []
        self.log_tau2 = []
        self.log_temp1 = []
        self.log_temp2 = []

        self._running = True

        def _control_loop():
            t_start = time.time()
            point_idx = 0

            while self._running and point_idx < n_points:
                loop_start = time.time()

                # Calcola angoli target per i motori (da radianti a gradi)
                q1_deg = np.rad2deg(q1_ctrl[point_idx])
                q2_deg = np.rad2deg(q2_ctrl[point_idx])

                # Calcola velocità massima per questo passo
                # (basata sulla velocità giunto pianificata)
                qd1_dps = abs(np.rad2deg(qd1_ctrl[point_idx]))
                qd2_dps = abs(np.rad2deg(qd2_ctrl[point_idx]))

                max_speed1 = max(10, min(int(qd1_dps * 1.5),
                                         self.config.max_motor_speed_dps))
                max_speed2 = max(10, min(int(qd2_dps * 1.5),
                                         self.config.max_motor_speed_dps))

                # Invia comandi di posizione assoluta
                self.motor1.absolute_position_control(q1_deg, max_speed1)
                self.motor2.absolute_position_control(q2_deg, max_speed2)

                # Monitoraggio stato (ogni 10 cicli per non sovraccaricare il bus)
                if self.config.enable_status_monitoring and point_idx % 10 == 0:
                    s1 = self.motor1.read_status_2()
                    s2 = self.motor2.read_status_2()

                    # Controlli di sicurezza
                    if self.config.enable_safety_checks:
                        if (s1.temperature > self.config.max_motor_temp or
                                s2.temperature > self.config.max_motor_temp):
                            print(f"\n   ALLARME: Sovratemperatura! "
                                  f"M1={s1.temperature}°C, M2={s2.temperature}°C")
                            self.emergency_stop()
                            return

                    # Log
                    self.log_time.append(t_ctrl[point_idx])
                    self.log_q1_cmd.append(q1_deg)
                    self.log_q2_cmd.append(q2_deg)
                    self.log_q1_actual.append(s1.angle)
                    self.log_q2_actual.append(s2.angle)
                    self.log_tau1.append(s1.torque_current)
                    self.log_tau2.append(s2.torque_current)
                    self.log_temp1.append(s1.temperature)
                    self.log_temp2.append(s2.temperature)

                point_idx += 1

                # Mantieni timing preciso
                elapsed = time.time() - loop_start
                sleep_time = dt_ctrl - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            self._running = False
            t_elapsed = time.time() - t_start
            print(f"\n   Traiettoria completata in {t_elapsed:.2f}s "
                  f"(pianificata: {t_ctrl[-1]:.2f}s)")

        if blocking:
            _control_loop()
        else:
            self._control_thread = threading.Thread(target=_control_loop)
            self._control_thread.daemon = True
            self._control_thread.start()

        return not self._emergency_stop

    def home(self, speed_dps: int = 100):
        """
        Porta il robot alla posizione home (0°, 0°).

        Args:
            speed_dps: Velocità di ritorno in dps
        """
        print("\n   Ritorno a home (0°, 0°)...")
        self.motor1.absolute_position_control(0.0, speed_dps)
        self.motor2.absolute_position_control(0.0, speed_dps)

    def plot_execution_log(self, save_path: str = None):
        """Grafico log esecuzione (comando vs effettivo)."""
        import matplotlib.pyplot as plt

        if not self.log_time:
            print("   Nessun log disponibile")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle('SCARA Controller - Log Esecuzione', fontsize=12,
                     fontweight='bold')

        t = self.log_time

        # Posizione giunto 1
        axes[0, 0].plot(t, self.log_q1_cmd, 'b-', label='Comando', linewidth=1.5)
        axes[0, 0].plot(t, self.log_q1_actual, 'r--', label='Effettivo', linewidth=1)
        axes[0, 0].set_ylabel('Angolo [°]')
        axes[0, 0].set_title('Giunto 1 - Posizione')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Posizione giunto 2
        axes[0, 1].plot(t, self.log_q2_cmd, 'b-', label='Comando', linewidth=1.5)
        axes[0, 1].plot(t, self.log_q2_actual, 'r--', label='Effettivo', linewidth=1)
        axes[0, 1].set_ylabel('Angolo [°]')
        axes[0, 1].set_title('Giunto 2 - Posizione')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Correnti
        axes[1, 0].plot(t, self.log_tau1, 'b-', label='Motore 1', linewidth=1.5)
        axes[1, 0].plot(t, self.log_tau2, 'r-', label='Motore 2', linewidth=1.5)
        axes[1, 0].set_xlabel('Tempo [s]')
        axes[1, 0].set_ylabel('Corrente [A]')
        axes[1, 0].set_title('Corrente Motori')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Temperature
        axes[1, 1].plot(t, self.log_temp1, 'b-', label='Motore 1', linewidth=1.5)
        axes[1, 1].plot(t, self.log_temp2, 'r-', label='Motore 2', linewidth=1.5)
        axes[1, 1].axhline(y=80, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Tempo [s]')
        axes[1, 1].set_ylabel('Temperatura [°C]')
        axes[1, 1].set_title('Temperatura Motori')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"   Log salvato: {save_path}")

        plt.close()
        return fig


# =============================================================================
# MAIN - INTEGRAZIONE SIMULATORE + CONTROLLORE
# =============================================================================

def run_pick_and_place(dry_run: bool = True, bus_type: str = "can",
                       channel: str = "can0", serial_port: str = "/dev/ttyUSB0",
                       motor1_id: int = 1, motor2_id: int = 2):
    """
    Esegui ciclo completo pick & place: simulazione + controllo motori.

    Args:
        dry_run: Se True, usa motori simulati (senza hardware)
        bus_type: Tipo di bus ("can" o "rs485")
        channel: Canale CAN
        serial_port: Porta seriale RS485
        motor1_id: ID motore giunto 1
        motor2_id: ID motore giunto 2
    """
    print("=" * 70)
    print("  SCARA CONTROLLER - PICK & PLACE")
    print("=" * 70)

    # --- 1. Configurazione robot e traiettoria ---

    params = SCARAParams()
    ARM_CONFIG = ArmConfig.LEFT

    constraints = TrajectoryConstraints(
        max_velocity=60.0,
        max_acceleration=120.0,
        profile_type=VelocityProfile.TRAPEZOIDAL,
        dt=0.001
    )

    waypoints = [
        Waypoint(x=75, y=15, dwell=0, payload=0),
        Waypoint(x=50, y=55, dwell=0.2, payload=None),
        Waypoint(x=35, y=60, dwell=0.5, payload=300),
        Waypoint(x=50, y=50, dwell=0, payload=None),
        Waypoint(x=70, y=40, dwell=0.2, payload=None),
        Waypoint(x=75, y=35, dwell=0.5, payload=0),
        Waypoint(x=75, y=15, dwell=0, payload=None),
    ]

    print(f"\n   Robot: {params.L1*100:.0f} + {params.L2*100:.0f} cm")
    print(f"   Modo: {'DRY RUN (simulato)' if dry_run else 'HARDWARE'}")

    # --- 2. Genera traiettoria con il simulatore ---

    print(f"\n{'='*70}")
    print("  FASE 1: PIANIFICAZIONE TRAIETTORIA")
    print(f"{'='*70}")

    t, q, qd, qdd, cart, success = generate_trajectory_with_payload(
        waypoints, params, ARM_CONFIG, constraints
    )

    if not success:
        print("\n   Traiettoria non valida!")
        return False

    # Simulazione dinamica
    tau, M_trace = simulate_trajectory(t, q, qd, qdd, cart['payload'], params)
    feasibility = analyze_feasibility(tau, params)

    total_dwell = sum(wp.dwell for wp in waypoints)
    print(f"\n   Tempo ciclo: {t[-1]:.2f} s")
    print(f"   Coppia max: {max(feasibility['tau1_peak'], feasibility['tau2_peak']):.3f} N*m")
    print(f"   Motori: {'OK' if feasibility['peak_ok'] else 'SUPERATO'}")

    if not feasibility['peak_ok']:
        print("\n   ATTENZIONE: Coppia motore superata! Continuare?")
        # In produzione, aggiungere conferma utente

    # Salva grafici simulazione
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plot_results(t, q, qd, qdd, tau, cart, M_trace, params, ARM_CONFIG,
                       constraints, feasibility, waypoints)
    sim_path = '/mnt/user-data/outputs/scara_pick_place.png'
    fig.savefig(sim_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Grafici simulazione: {sim_path}")

    # --- 3. Setup motori ---

    print(f"\n{'='*70}")
    print("  FASE 2: CONFIGURAZIONE MOTORI")
    print(f"{'='*70}")

    if dry_run:
        motor1 = SimulatedMotor(motor1_id)
        motor2 = SimulatedMotor(motor2_id)
        print("\n   Motori simulati creati")
    else:
        bus_enum = BusInterface.CAN if bus_type == "can" else BusInterface.RS485
        protocol = MotorProtocol(
            bus_type=bus_enum,
            channel=channel,
            serial_port=serial_port
        )
        protocol.connect()
        motor1 = V3Motor(motor1_id, protocol, gear_ratio=params.gear_ratio)
        motor2 = V3Motor(motor2_id, protocol, gear_ratio=params.gear_ratio)

    controller_config = ControllerConfig(
        control_rate_hz=100.0,
        max_motor_speed_dps=720,
        position_accel_dps2=10000,
        position_decel_dps2=10000,
        comm_timeout_ms=500,
    )

    controller = TrajectoryController(motor1, motor2, controller_config)

    if not controller.initialize():
        print("\n   Errore inizializzazione motori!")
        return False

    # --- 4. Esecuzione traiettoria ---

    print(f"\n{'='*70}")
    print("  FASE 3: ESECUZIONE TRAIETTORIA")
    print(f"{'='*70}")

    success = controller.execute_trajectory(t, q, qd, cart, blocking=True)

    if success:
        print(f"\n   Ciclo pick & place completato!")
    else:
        print(f"\n   Errore durante esecuzione!")

    # --- 5. Salva log esecuzione ---

    log_path = '/mnt/user-data/outputs/scara_controller_log.png'
    controller.plot_execution_log(save_path=log_path)

    # --- 6. Shutdown ---

    print(f"\n{'='*70}")
    print("  FASE 4: SHUTDOWN")
    print(f"{'='*70}")

    controller.shutdown()

    if not dry_run:
        protocol.disconnect()

    print(f"\n{'='*70}")
    print(f"  COMPLETATO")
    print(f"{'='*70}\n")

    return success


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SCARA Robot Trajectory Controller per motori MyActuator X-V3'
    )
    parser.add_argument('--dry-run', action='store_true', default=True,
                        help='Esegui senza hardware (motori simulati)')
    parser.add_argument('--hardware', action='store_true',
                        help='Esegui con hardware reale')
    parser.add_argument('--bus', choices=['can', 'rs485'], default='can',
                        help='Tipo di bus comunicazione')
    parser.add_argument('--channel', default='can0',
                        help='Canale CAN (default: can0)')
    parser.add_argument('--serial-port', default='/dev/ttyUSB0',
                        help='Porta seriale RS485 (default: /dev/ttyUSB0)')
    parser.add_argument('--motor1-id', type=int, default=1,
                        help='ID motore giunto 1 (default: 1)')
    parser.add_argument('--motor2-id', type=int, default=2,
                        help='ID motore giunto 2 (default: 2)')

    args = parser.parse_args()

    dry_run = not args.hardware

    run_pick_and_place(
        dry_run=dry_run,
        bus_type=args.bus,
        channel=args.channel,
        serial_port=args.serial_port,
        motor1_id=args.motor1_id,
        motor2_id=args.motor2_id,
    )
