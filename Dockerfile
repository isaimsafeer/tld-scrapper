FROM python:3.13-slim

# Install base system dependencies
RUN apt-get update && apt-get install -y \
    wget unzip curl gnupg ca-certificates \
    libasound2 libatk-bridge2.0-0 libatk1.0-0 libcups2 libdbus-1-3 \
    libdrm2 libgbm1 libgtk-3-0 libnspr4 libnss3 libxcomposite1 libxdamage1 \
    libxfixes3 libxkbcommon0 libxrandr2 libxshmfence1 fonts-liberation xdg-utils \
    && rm -rf /var/lib/apt/lists/*

################################################################################
# Install Chrome-for-Testing (Browser + Matching ChromeDriver)
################################################################################

# Get the latest stable version of Chrome-for-Testing
RUN CFT_VERSION=$(curl -s https://googlechromelabs.github.io/chrome-for-testing/LATEST_RELEASE_STABLE) \
    && echo "Using Chrome for Testing version: $CFT_VERSION" \
    \
    # Download Chrome browser
    && wget -q "https://storage.googleapis.com/chrome-for-testing-public/${CFT_VERSION}/linux64/chrome-linux64.zip" -O /tmp/chrome.zip \
    && unzip /tmp/chrome.zip -d /opt \
    && mv /opt/chrome-linux64 /opt/chrome \
    \
    # Download matching ChromeDriver
    && wget -q "https://storage.googleapis.com/chrome-for-testing-public/${CFT_VERSION}/linux64/chromedriver-linux64.zip" -O /tmp/chromedriver.zip \
    && unzip /tmp/chromedriver.zip -d /opt \
    && mv /opt/chromedriver-linux64/chromedriver /usr/local/bin/chromedriver \
    && chmod +x /usr/local/bin/chromedriver \
    \
    # Cleanup
    && rm -rf /tmp/*.zip /opt/chromedriver-linux64

# Set Chrome binary path for Selenium
ENV CHROME_BIN=/opt/chrome/chrome
ENV CHROMEDRIVER_PATH=/usr/local/bin/chromedriver

################################################################################
# Python app setup
################################################################################

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
