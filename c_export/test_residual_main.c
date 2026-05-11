#include "cptservo_residual.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Read one whitespace-delimited token into out (size cap=63). Returns 1 on
 * success, 0 on EOF/no-token.  Uses strtod for robust NaN/Inf parsing. */
static int read_token(double *out)
{
    char buf[64];
    int idx = 0;
    int ch;
    /* skip leading whitespace */
    do {
        ch = getchar();
        if (ch == EOF) {
            return 0;
        }
    } while (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r');
    do {
        if (idx < (int)(sizeof(buf) - 1)) {
            buf[idx++] = (char)ch;
        }
        ch = getchar();
    } while (ch != EOF && ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r');
    buf[idx] = '\0';
    /* Lowercase for portable nan/inf token detection (TCC's strtod does not
     * parse these on Windows). */
    char lower[64];
    for (int i = 0; i < idx; i++) {
        lower[i] = (char)tolower((unsigned char)buf[i]);
    }
    lower[idx] = '\0';
    if (strncmp(lower, "nan", 3) == 0) {
        *out = NAN;
        return 1;
    }
    if (strcmp(lower, "inf") == 0 || strcmp(lower, "infinity") == 0
        || strcmp(lower, "+inf") == 0) {
        *out = INFINITY;
        return 1;
    }
    if (strcmp(lower, "-inf") == 0 || strcmp(lower, "-infinity") == 0) {
        *out = -INFINITY;
        return 1;
    }
    char *endp = NULL;
    *out = strtod(buf, &endp);
    if (endp == buf) {
        return 0;
    }
    return 1;
}

int main(void)
{
    cpt_controller_t ctl;
    double error;
    double T_K;
    double B_uT;
    double I_norm;

    cpt_controller_init(&ctl);

    while (read_token(&error)
        && read_token(&T_K)
        && read_token(&B_uT)
        && read_token(&I_norm)) {
        const double rf = cpt_controller_step(&ctl, error, T_K, B_uT, I_norm);
        printf("%.17g\n", rf);
    }

    return 0;
}
