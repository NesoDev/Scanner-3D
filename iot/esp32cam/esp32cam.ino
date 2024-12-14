#include "esp_camera.h"
#include "WiFi.h"
#include "HTTPClient.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

const char* ssid = "NESO_WIFI_2.4";
const char* password = "print(\"hola_mundo\")";
const char* backendUrl = "http://192.168.1.101:8000/scan/stream";

void startCameraServer();

void setup() {
    Serial.begin(115200);
    
    // Asegurar que el flash está apagado
    pinMode(4, OUTPUT);
    digitalWrite(4, LOW);
    
    // Conexión WiFi
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi conectado");
    
    // Configuración de la cámara
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
    
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Error al inicializar la cámara: 0x%x\n", err);
        return;
    }
    
    // Notificar al backend
    HTTPClient http;
    http.begin(backendUrl);
    http.addHeader("Content-Type", "application/json");
    String jsonData = "{\"ip_cam\": \"" + WiFi.localIP().toString() + "\"}";
    http.POST(jsonData);
    http.end();
    
    // Iniciar servidor de streaming
    startCameraServer();
    Serial.printf("Servidor de streaming iniciado en http://%s:80/stream\n", WiFi.localIP().toString().c_str());
}

void loop() {
    // Watchdog para WiFi
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("Reconectando WiFi...");
        WiFi.reconnect();
    }
    delay(100);
}