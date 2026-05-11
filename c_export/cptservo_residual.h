#ifndef CPTSERVO_RESIDUAL_H
#define CPTSERVO_RESIDUAL_H

#include "cptservo_coeffs.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double x[2];
    double h[CPT_HIDDEN_SIZE];
    double I_hat;
    double last_T_K;
    double last_B_uT;
    double err_window[CPT_ERR_WINDOW_SIZE];
    uint32_t err_index;
    double err_integral;
    double last_error;
    double last_rf;
    double last_delta_rf;
} cpt_controller_t;

void cpt_controller_init(cpt_controller_t *ctl);
double cpt_controller_step(
    cpt_controller_t *ctl,
    double error,
    double T_K,
    double B_uT,
    double I_norm
);

#ifdef __cplusplus
}
#endif

#endif /* CPTSERVO_RESIDUAL_H */
