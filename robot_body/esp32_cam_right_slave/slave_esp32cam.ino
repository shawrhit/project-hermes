#include "WiFi.h"
#include "esp_camera.h"
#include "HTTPClient.h"
#include "freertos/queue.h"

// -------- Wi-Fi + Server --------
#define WIFI_SSID "RoboticsLab"
#define WIFI_PASS "robot@123"
#define SERVER_URL "http://192.168.137.1:5000/upload?cam=B"

// -------- Trigger input from master --------
#define TRIGGER_IN_PIN 13

// -------- Camera Config (AI Thinker) --------
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// -------- Globals --------
WiFiClient client;
HTTPClient http;
QueueHandle_t fbQueue;

// ========================================================
// UPLOAD TASK — runs in background on Core 1
// ========================================================
void uploadTask(void *pv) {
  camera_fb_t * fb;
  while (true) {
    if (xQueueReceive(fbQueue, &fb, portMAX_DELAY) == pdTRUE) {
      int code = http.POST(fb->buf, fb->len);
      if (code != 200) {
        http.end();
        http.begin(client, SERVER_URL);
        http.addHeader("Content-Type", "image/jpeg");
      }
      esp_camera_fb_return(fb);
    }
  }
}

// ========================================================
// SETUP
// ========================================================
void setup() {
  Serial.begin(115200);
  pinMode(TRIGGER_IN_PIN, INPUT);

  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(200);
    Serial.print(".");
  }
  Serial.printf("\nConnected, IP: %s\n", WiFi.localIP().toString().c_str());

  // ---- Camera Config ----
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

  // ---- Depth pipeline settings ----
  config.frame_size = FRAMESIZE_VGA;    // 640×480
  config.jpeg_quality = 20;             // balance detail/speed
  config.fb_count = 2;                  // double buffer for async

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed 0x%x\n", err);
    while (1) delay(1000);
  }

  http.begin(client, SERVER_URL);
  http.addHeader("Content-Type", "image/jpeg");

  fbQueue = xQueueCreate(2, sizeof(camera_fb_t *));
  xTaskCreatePinnedToCore(uploadTask, "upload", 8192, NULL, 1, NULL, 1);

  Serial.println("Slave ready. Waiting for trigger...");
}

// ========================================================
// LOOP — waits for trigger pulse, then captures and queues
// ========================================================
void loop() {
  static uint64_t lastTrig = 0;

  if (digitalRead(TRIGGER_IN_PIN) == HIGH) {
    // rising edge detected
    if (esp_timer_get_time() - lastTrig > 10000) {  // debounce 10 ms
      lastTrig = esp_timer_get_time();

      camera_fb_t * fb = esp_camera_fb_get();
      if (!fb) return;

      xQueueSend(fbQueue, &fb, 0);
      Serial.printf("Captured %d bytes\n", fb->len);
    }
  }

  // light delay to yield CPU
  delay(1);
}
