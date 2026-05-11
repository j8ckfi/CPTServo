#include "cptservo_residual.h"

#include <math.h>

static double clip_double(double x, double lo, double hi)
{
    if (x < lo) {
        return lo;
    }
    if (x > hi) {
        return hi;
    }
    return x;
}

static double bounded_sensor(double raw_value, double last_good, double nominal, double max_step)
{
    double delta;
    double bounded;

    if (!isfinite(raw_value)) {
        return last_good;
    }
    delta = clip_double(raw_value - last_good, -max_step, max_step);
    bounded = last_good + delta;
    if (!isfinite(bounded)) {
        return nominal;
    }
    return bounded;
}

static double softplus_clipped(double x)
{
    const double clipped = clip_double(x, -40.0, 40.0);
    /* softplus(x) = log(1+exp(x)).  Use log(1+exp) rather than log1p(exp) so
     * the C source compiles against C runtimes that omit log1p (e.g. TCC on
     * Windows).  Within the clipped range, the values agree to last ULP. */
    return log(1.0 + exp(clipped));
}

static double lqr_step(cpt_controller_t *ctl, double error)
{
    double raw;
    const double k0 = LQR_K[0];
    const double k1 = LQR_K[1];

    ctl->x[0] = error;
    ctl->x[1] += ctl->x[0] * LQR_DT;

    raw = k0 * ctl->x[0] + k1 * ctl->x[1];
    if (raw > LQR_RF_LIMIT) {
        raw = LQR_RF_LIMIT;
        if (fabs(k1) > 1.0e-30) {
            ctl->x[1] = (raw - k0 * ctl->x[0]) / k1;
        }
    } else if (raw < -LQR_RF_LIMIT) {
        raw = -LQR_RF_LIMIT;
        if (fabs(k1) > 1.0e-30) {
            ctl->x[1] = (raw - k0 * ctl->x[0]) / k1;
        }
    }

    return raw;
}

static void build_features(
    cpt_controller_t *ctl,
    double error,
    double raw_T_K,
    double raw_B_uT,
    double raw_I_norm,
    double phi[CPT_FEATURE_DIM]
)
{
    const double T_K = bounded_sensor(
        raw_T_K,
        ctl->last_T_K,
        CPT_T_NOM_K,
        CPT_MAX_SENSOR_STEP_T_K
    );
    const double B_uT = bounded_sensor(
        raw_B_uT,
        ctl->last_B_uT,
        CPT_B_NOM_UT,
        CPT_MAX_SENSOR_STEP_B_UT
    );
    const double I_env = bounded_sensor(
        raw_I_norm,
        ctl->I_hat,
        CPT_I_NOM,
        CPT_MAX_SENSOR_STEP_I_NORM
    );
    const double prev_I_hat = ctl->I_hat;
    const double tau_floor = CPT_INTENSITY_TAU_S > 1.0e-9 ? CPT_INTENSITY_TAU_S : 1.0e-9;
    const double alpha = fmin(1.0, CPT_CONTROL_DT_S / tau_floor);
    double dT_dt;
    double dB_dt;
    double dI_dt;
    double sum = 0.0;
    double sumsq = 0.0;
    double mean;
    double variance;
    double std;
    uint32_t i;

    ctl->I_hat += alpha * (I_env - ctl->I_hat);
    dT_dt = (T_K - ctl->last_T_K) / CPT_CONTROL_DT_S;
    dB_dt = (B_uT - ctl->last_B_uT) / CPT_CONTROL_DT_S;
    dI_dt = (ctl->I_hat - prev_I_hat) / CPT_CONTROL_DT_S;
    ctl->last_T_K = T_K;
    ctl->last_B_uT = B_uT;

    ctl->err_integral = clip_double(
        ctl->err_integral + error * CPT_CONTROL_DT_S,
        -CPT_ERR_INTEGRAL_CLIP,
        CPT_ERR_INTEGRAL_CLIP
    );
    ctl->err_window[ctl->err_index] = error;
    ctl->err_index = (ctl->err_index + 1U) % (uint32_t)CPT_ERR_WINDOW_SIZE;

    for (i = 0U; i < (uint32_t)CPT_ERR_WINDOW_SIZE; ++i) {
        const double v = ctl->err_window[i];
        sum += v;
        sumsq += v * v;
    }
    mean = sum / (double)CPT_ERR_WINDOW_SIZE;
    variance = sumsq / (double)CPT_ERR_WINDOW_SIZE - mean * mean;
    if (variance < 0.0) {
        variance = 0.0;
    }
    std = sqrt(variance);

    phi[0] = (T_K - CPT_T_NOM_K) / 10.0;
    phi[1] = (B_uT - CPT_B_NOM_UT) / 10.0;
    phi[2] = ctl->I_hat - CPT_I_NOM;
    phi[3] = dT_dt / 10.0;
    phi[4] = dB_dt / 10.0;
    phi[5] = dI_dt;
    phi[6] = error * 1.0e3;
    phi[7] = (error - ctl->last_error) * 1.0e3;
    phi[8] = ctl->err_integral * 1.0e6;
    phi[9] = mean * 1.0e3;
    phi[10] = std * 1.0e3;
    phi[11] = ctl->last_rf / fmax(CPT_RF_LIMIT_HZ, 1.0e-9);
    phi[12] = ctl->last_delta_rf / 10.0;
    phi[13] = 1.0;
}

static void cfc_step(cpt_controller_t *ctl, const double phi[CPT_FEATURE_DIM])
{
    double next_h[CPT_HIDDEN_SIZE];
    uint32_t i;
    uint32_t j;

    for (i = 0U; i < (uint32_t)CPT_HIDDEN_SIZE; ++i) {
        double cand_raw = CFC_B_C[i];
        double tau_raw = CFC_B_TAU[i];
        double tau;
        double gate;

        for (j = 0U; j < (uint32_t)CPT_FEATURE_DIM; ++j) {
            cand_raw += CFC_W_C[i][j] * phi[j];
            tau_raw += CFC_W_TAU[i][j] * phi[j];
        }
        for (j = 0U; j < (uint32_t)CPT_HIDDEN_SIZE; ++j) {
            cand_raw += CFC_U_C[i][j] * ctl->h[j];
            tau_raw += CFC_U_TAU[i][j] * ctl->h[j];
        }

        tau = softplus_clipped(tau_raw) + 1.0e-3;
        gate = exp(-CPT_CONTROL_DT_S / tau);
        next_h[i] = gate * ctl->h[i] + (1.0 - gate) * tanh(cand_raw);
    }

    for (i = 0U; i < (uint32_t)CPT_HIDDEN_SIZE; ++i) {
        ctl->h[i] = next_h[i];
    }
}

static double readout(const cpt_controller_t *ctl, const double phi[CPT_FEATURE_DIM])
{
    double raw = CFC_B_OUT[0];
    uint32_t i;

    for (i = 0U; i < (uint32_t)CPT_HIDDEN_SIZE; ++i) {
        raw += CFC_W_OUT[i] * ctl->h[i];
    }
    if (CPT_OUTPUT_FEATURE_SKIP != 0) {
        for (i = 0U; i < (uint32_t)CPT_FEATURE_DIM; ++i) {
            raw += CFC_W_FEAT_OUT[i] * phi[i];
        }
    }
    return raw;
}

static void commit_action(cpt_controller_t *ctl, double error, double rf)
{
    const double clipped_rf = clip_double(rf, -CPT_RF_LIMIT_HZ, CPT_RF_LIMIT_HZ);
    ctl->last_delta_rf = clipped_rf - ctl->last_rf;
    ctl->last_rf = clipped_rf;
    ctl->last_error = error;
}

void cpt_controller_init(cpt_controller_t *ctl)
{
    uint32_t i;

    if (ctl == NULL) {
        return;
    }

    ctl->x[0] = 0.0;
    ctl->x[1] = 0.0;
    for (i = 0U; i < (uint32_t)CPT_HIDDEN_SIZE; ++i) {
        ctl->h[i] = 0.0;
    }
    ctl->I_hat = CPT_I_NOM;
    ctl->last_T_K = CPT_T_NOM_K;
    ctl->last_B_uT = CPT_B_NOM_UT;
    for (i = 0U; i < (uint32_t)CPT_ERR_WINDOW_SIZE; ++i) {
        ctl->err_window[i] = 0.0;
    }
    ctl->err_index = 0U;
    ctl->err_integral = 0.0;
    ctl->last_error = 0.0;
    ctl->last_rf = 0.0;
    ctl->last_delta_rf = 0.0;
}

double cpt_controller_step(
    cpt_controller_t *ctl,
    double error,
    double T_K,
    double B_uT,
    double I_norm
)
{
    double phi[CPT_FEATURE_DIM];
    double u_lqr;
    double raw;
    double rf;

    if (ctl == NULL) {
        return 0.0;
    }

    u_lqr = 0.0;
    if (CPT_RESIDUAL_MODE != 0) {
        u_lqr = lqr_step(ctl, error);
    }

    build_features(ctl, error, T_K, B_uT, I_norm, phi);
    cfc_step(ctl, phi);
    raw = readout(ctl, phi);

    if (CPT_RESIDUAL_MODE != 0) {
        const double residual = clip_double(
            raw,
            -CPT_RESIDUAL_LIMIT_HZ,
            CPT_RESIDUAL_LIMIT_HZ
        );
        rf = clip_double(u_lqr + residual, -CPT_RF_LIMIT_HZ, CPT_RF_LIMIT_HZ);
    } else {
        rf = clip_double(raw, -CPT_RF_LIMIT_HZ, CPT_RF_LIMIT_HZ);
    }

    commit_action(ctl, error, rf);
    return rf;
}
