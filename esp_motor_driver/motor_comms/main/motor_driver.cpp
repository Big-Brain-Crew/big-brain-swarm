#include "motor_driver.hpp"

MotorDriver::MotorDriver(const MotorDriver::Config &config)
  : en_left(config.ENABLE_L),
    en_right(config.ENABLE_R),
    left_a(config.IN1), 
    left_b(config.IN2),
    right_a(config.IN3),
    right_b(config.IN4)
    {  
        // make bit mask of config gpio pins
        auto GPIO_OUTPUT_PIN_SEL  ((1ULL<<left_a) | (1ULL<<left_b) | (1ULL<<right_a) | (1ULL<<right_b));
        //zero-initialize the config structure.
        gpio_config_t io_conf = {};
        //disable interrupt
        io_conf.intr_type = GPIO_INTR_DISABLE;
        //set as output mode
        io_conf.mode = GPIO_MODE_OUTPUT;
        //bit mask of the pins that you want to set,e.g.GPIO18/19
        io_conf.pin_bit_mask = GPIO_OUTPUT_PIN_SEL;
        //disable pull-down mode
        io_conf.pull_down_en = (gpio_pulldown_t)0;
        //disable pull-up mode
        io_conf.pull_up_en = (gpio_pullup_t)0;
        //configure GPIO with the given settings
        gpio_config(&io_conf);


        /*
        * Prepare and set configuration of timers
        * that will be used by Motor Controller
        */
        ledc_timer_config_t motor_driver_timer = {
            .speed_mode = LEDC_LOW_SPEED_MODE,    // timer mode
            .duty_resolution = LEDC_TIMER_13_BIT, // resolution of PWM duty
            .timer_num = LEDC_TIMER_0,            // timer index
            .freq_hz = 500,                       // frequency of PWM signal
            .clk_cfg = LEDC_AUTO_CLK,             // Auto select the source clock
        };
        // Set configuration of timer0
        ledc_timer_config(&motor_driver_timer);

        // Left PWM
        ledc_channel_config_t motor_driver_L_channel ={
            .gpio_num = en_left,
            .speed_mode = LEDC_LOW_SPEED_MODE,
            .channel = LEDC_CHANNEL_0,
            .intr_type = LEDC_INTR_DISABLE,
            .timer_sel = LEDC_TIMER_0,
            .duty = 0,
            .hpoint = 0
        };
        
        // Right_PWM
        ledc_channel_config_t motor_driver_R_channel ={
            .gpio_num = en_right,
            .speed_mode = LEDC_LOW_SPEED_MODE,
            .channel = LEDC_CHANNEL_1,
            .intr_type = LEDC_INTR_DISABLE,
            .timer_sel = LEDC_TIMER_0,
            .duty = 0,
            .hpoint = 0
        };

        // init the pwm channels for the L and R en pins
        ledc_channel_config(&motor_driver_L_channel);
        ledc_channel_config(&motor_driver_R_channel);
    }

MotorDriver::~MotorDriver(){
    stop();
}

void MotorDriver::stop(){
    disable_left();
    disable_right();   
}

void MotorDriver::left_side(int pwm){
    // accounting for negative pwm ranges
    bool left_dir = FORWARD;
    if (pwm < 0){
        pwm *= -1;
        left_dir = BACKWARD;
    }
    enable_left(left_dir);

    // pwm that shiet
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, pwm_to_bits(pwm, 13));
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0);
}

void MotorDriver::right_side(int pwm){
    // accounting for negative pwm ranges
    bool right_dir = FORWARD;
    if (pwm < 0){
        pwm *= -1;
        right_dir = BACKWARD;
    }
    enable_right(right_dir);
    // pwm that sheit
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_1, pwm_to_bits(pwm, 13));
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_1);
}

void MotorDriver::enable_left(bool dir){
    // left_motor_direction is checking which is the forward direction
    //   dir is the command to drive forward or backwards.
    gpio_set_level(left_a, left_motor_direction? dir:!dir);
    gpio_set_level(left_b, left_motor_direction? !dir:dir);
}

void MotorDriver::enable_right(bool dir){
    // right_motor_direction is checking which is the forward direction
    //   dir is the command to drive forward or backwards.
    gpio_set_level(right_a, right_motor_direction? dir:!dir);
    gpio_set_level(right_b, right_motor_direction? !dir:dir);
}

// just for configuration not needed for use.  System defaults both to FORWARD
void MotorDriver::set_motor_directions(bool left_dir, bool right_dir){
    left_motor_direction = left_dir;
    right_motor_direction = right_dir;
}

void MotorDriver::drivelr(int left, int right){
    left_side(left);
    right_side(right);
}

void MotorDriver::disable_left(){
    gpio_set_level(left_a, 0);
    gpio_set_level(left_b, 0);
    drivelr(0,0);
}

void MotorDriver::disable_right(){
    gpio_set_level(right_a, 0);
    gpio_set_level(right_b, 0);
    drivelr(0,0);
}

// input dp and the bit resolution of the PWM you are converting to
// example, 50% at 13bit resolution will return 4096
uint32_t MotorDriver::pwm_to_bits(int duty_percentage, int bit_resolution){
    float max = (float) pow(2, bit_resolution);
    float pwmf = std::abs(duty_percentage);
    uint32_t duty = (int) ((duty_percentage/100.0f) * max);

    return duty;
}