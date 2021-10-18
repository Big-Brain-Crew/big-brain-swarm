#include <stdio.h>
#include <math.h>
#include "driver/gpio.h"
#include "driver/ledc.h"

#pragma once

// defines for standardizing forward and backward
#define FORWARD    1
#define BACKWARD   0

// pub
class MotorDriver{
    
    public:
    // pin Definitions for controlling the l298n
    struct Config
    {
        gpio_num_t ENABLE_L;
        gpio_num_t ENABLE_R;
        gpio_num_t IN1;
        gpio_num_t IN2;
        gpio_num_t IN3;
        gpio_num_t IN4;
    };

    MotorDriver(const Config &config);

    ~MotorDriver();

    void stop();

    // flip motor
    void set_motor_directions(bool left_dir, bool right_dir);

    // drive motors arbitrary values
    // -100 to 100 (dp for pwm)
    void drivelr(int left, int right);

    uint32_t pwm_to_bits(int duty_percentage, int bit_resolution);

// priv
    // GPIO defs
    gpio_num_t en_left;
    gpio_num_t en_right;
    gpio_num_t left_a;
    gpio_num_t left_b;
    gpio_num_t right_a;
    gpio_num_t right_b;

    // directionality for each motor
    bool left_motor_direction{FORWARD};
    bool right_motor_direction{FORWARD};

    // each side's controls
    void left_side(int pwm);
    void right_side(int pwm);

    // enable
    void enable_left(bool dir);
    void enable_right(bool dir);

    // disable
    void disable_left();
    void disable_right();
};