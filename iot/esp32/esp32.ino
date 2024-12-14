#include <WiFi.h>
#include <HTTPClient.h>
#include <WebServer.h>

// Configuración WiFi
const char* ssid = "NESO_WIFI_2.4";
const char* password = "print(\"hola_mundo\")";
const char* backendUrl = "http://192.168.1.101:8000/scan/pair";

// Configuración del servidor web
WebServer server(80);

// Define los pines del motor paso a paso
const int motorPin1 = 13;
const int motorPin2 = 12;
const int motorPin3 = 14;
const int motorPin4 = 27;

// Variables de control del motor
bool isMotorRunning = false;
const int STEPS_PER_REVOLUTION = 502;
const int STEP_DELAY = 10; // Milisegundos entre pasos
int currentStep = 0; // Para tracking del progreso

String connectWifi() {
    Serial.print("Conectando a Wi-Fi...");
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi conectado");
    IPAddress ip = WiFi.localIP();
    Serial.print("IP address: ");
    Serial.println(ip);
    return ip.toString();
}

void sendIp(HTTPClient& http, const String& ip) {
    Serial.println("Enviando IP al backend...");
    http.begin(backendUrl);
    http.addHeader("Content-Type", "application/json");
    String jsonData = "{\"ip_esp32\": \"" + ip + "\"}";
    int httpCode = http.POST(jsonData);

    if (httpCode > 0) {
        Serial.printf("Código de respuesta: %d\n", httpCode);
        if (httpCode == HTTP_CODE_OK) {
            String payload = http.getString();
            Serial.println("Respuesta del servidor:");
            Serial.println(payload);
        }
    } else {
        Serial.printf("Error en la solicitud HTTP: %s\n", http.errorToString(httpCode).c_str());
    }
    http.end();
}

void stepForward() {
    // Secuencia para girar en sentido horario
    digitalWrite(motorPin1, HIGH);
    digitalWrite(motorPin2, LOW);
    digitalWrite(motorPin3, LOW);
    digitalWrite(motorPin4, LOW);
    delay(10);

    digitalWrite(motorPin1, LOW);
    digitalWrite(motorPin2, HIGH);
    digitalWrite(motorPin3, LOW);
    digitalWrite(motorPin4, LOW);
    delay(10);

    digitalWrite(motorPin1, LOW);
    digitalWrite(motorPin2, LOW);
    digitalWrite(motorPin3, HIGH);
    digitalWrite(motorPin4, LOW);
    delay(10);

    digitalWrite(motorPin1, LOW);
    digitalWrite(motorPin2, LOW);
    digitalWrite(motorPin3, LOW);
    digitalWrite(motorPin4, HIGH);
    delay(10);
    
    delay(STEP_DELAY);
    currentStep++;
}

void handleMotorStart() {
    isMotorRunning = true;
    currentStep = 0;
    
    for (int i = 0; i < STEPS_PER_REVOLUTION; i++) {
        stepForward();
    }
    
    isMotorRunning = false;
    server.send(200, "text/plain", "Motor ha girado 360°");
}

void handleMotorStatus() {
    // Crear JSON manualmente
    String response = "{";
    response += "\"isRunning\":" + String(isMotorRunning ? "true" : "false") + ",";
    response += "\"totalSteps\":" + String(STEPS_PER_REVOLUTION) + ",";
    response += "\"currentStep\":" + String(currentStep) + ",";
    response += "\"progress\":" + String((currentStep * 100.0) / STEPS_PER_REVOLUTION);
    response += "}";
    
    server.send(200, "application/json", response);
}

void setup() {
    Serial.begin(115200);
    
    // Configurar pines del motor
    pinMode(motorPin1, OUTPUT);
    pinMode(motorPin2, OUTPUT);
    pinMode(motorPin3, OUTPUT);
    pinMode(motorPin4, OUTPUT);
    
    // Conectar a WiFi y enviar IP
    String ip = connectWifi();
    HTTPClient http;
    sendIp(http, ip);
    
    // Configurar rutas del servidor web
    server.on("/scan/start", HTTP_GET, handleMotorStart);
    server.on("/scan/status", HTTP_GET, handleMotorStatus);
    server.begin();
    
    Serial.println("Servidor HTTP iniciado");
}

void loop() {
    server.handleClient();
}