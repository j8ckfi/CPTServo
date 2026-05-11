CC ?= gcc
PYTHON ?= python
CFLAGS ?= -O2 -Wall -Wextra -Wpedantic -std=c11
LDLIBS ?= -lm

TARGET = test_residual
SRCS = cptservo_residual.c test_residual_main.c
HDRS = cptservo_residual.h cptservo_coeffs.h

.PHONY: all verify clean

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) $(LDLIBS)

verify: all
	cd .. && $(PYTHON) scripts/verify_c_residual.py

clean:
	rm -f $(TARGET) $(TARGET).exe *.o
