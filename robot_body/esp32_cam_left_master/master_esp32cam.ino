#include "WiFi.h"
#include "esp_camera.h"
#include "HTTPClient.h"
#include "freertos/queue.h"

// ----- CONFIG -----
#define WIFI_SSID "RoboticsLab"
#define WIFI_PASS "robot@123"
#define SERVER_URL "http://192.168.137.1:5000/upload?cam=A"
#define TRIGGER_OUT_PIN 12

#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

// ----- Globals -----
WiFiClient client;
HTTPClient http;
QueueHandle_t fbQueue;

// -----------------------------------------------
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

// -----------------------------------------------
void setup() {
  Serial.begin(115200);
  pinMode(TRIGGER_OUT_PIN, OUTPUT);
  digitalWrite(TRIGGER_OUT_PIN, LOW);

  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(200); }

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;  config.pin_d7 = Y9_GPIO_NUM;
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
  config.jpeg_quality = 20;
  config.fb_count = 2;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Camera init failed");
    while (1) delay(1000);
  }

  http.begin(client, SERVER_URL);
  http.addHeader("Content-Type", "image/jpeg");

  fbQueue = xQueueCreate(2, sizeof(camera_fb_t *));
  xTaskCreatePinnedToCore(uploadTask, "upload", 8192, NULL, 1, NULL, 1);

  Serial.println("Async capture/upload started.");
}

// -----------------------------------------------
void loop() {
  static unsigned long last = 0;
  digitalWrite(TRIGGER_OUT_PIN, HIGH);
  delayMicroseconds(5000);
  digitalWrite(TRIGGER_OUT_PIN, LOW);

  uint64_t t0 = esp_timer_get_time();
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) return;

  xQueueSend(fbQueue, &fb, 0);  // hand off to upload task

  float fps = 1000000.0 / (esp_timer_get_time() - t0 + 1);
  last = millis();
  Serial.printf("Captured %d bytes | %.1f FPS (instant)\n", fb->len, fps);

  // minimal pacing; keeps Wi-Fi stable
  delay(30);
}
