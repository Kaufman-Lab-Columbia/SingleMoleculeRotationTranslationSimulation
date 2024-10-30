///////////////////////////////////////////
////
//// Simultaneous Rotation And Translation simulation that outputs a .tif with random coordinates (Or grid, Not yet implemented) And a Gaussian PSF
//// Features both rotate And translate with user specified median timescales sampled on a gaussian distribution with optional dynamic exchange implementation
//// This version runs on Nvidia GPUs using CUDA for decreased calcluation time
////
//////////////////////////////////////////

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <direct.h>
#include <limits.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>

#define     HEIGHT  300                     // Height of output.txt
#define     WIDTH   300                     // Width of output.txt
#define     C       2.35482005              // FWHM with unit stdev
#define     PI	    3.14159265358979323     // circumference by double the radius
#define     DEC     8                       // number of decimal places in the output  

#define     init_pos                1       // How initial XY-positions are determined. (0) Random, (1) Grid
#define     feats                   10      // If using Random Initial Positions: Number of molecules to simulation
#define     separation              15      // If using Grid Initial Positions: How far apart (pixels) must molecules be on the grid

#define     frames                  10   // Number of frames to simulate
#define     pixel_size              166     // Pixel Size (nm)
#define     tbf                     0.5     // Time between frames (s)
#define     aperture_detect         0.95    // NA of objective lens
#define     PSF_fwhm                294     // Point Spread Function FWHM (nm)
#define     addExcitation           1       // Include Excitation Probility Contributions (0) No, (1) Yes

#define     photons_per_frame       300.0   // Photons per frame
#define     photomultiplier_gain    200.0   // Photomultiplier Gain (EM Gain)
#define     bkg_constant            0.00    // Background floor magnitude
#define     bkg_amp                 0.000   // Random Background noise amplitude

#define     ratio                   1.0     // Ratio of IPA:OPA dynamics; for anisotropy (1.0 means isotropic)

#define     D_med                   -10.0   // median of log_10(Dmsd) distribution
#define     D_fwhm                  0.0     // FWHM of log_10(Dmsd) distribution
#define     D_corr                  0.0     // correlation

#define     Dxch_med                100000  // median number of frames before D_exchange
#define     Dxch_fwhm               0.0     // FWHM of log_10(D_exchange) distribution
#define     Dxch_corr               0.0     // correlation

#define     tau_med                 1.0     // median of log_10(tauc) distribution
#define     tau_fwhm                2.0     // FWHM of log_10(tauc) distribution
#define     tau_corr                0.0     // Correlation of tau to previous timescales (rho_r in Kevin's paper; fast remains fast and slow remains slow)

#define     xch_med                 0.00    // Exchange Method:  (1) median number of frames before tau_exchange (2) median log tau_ex
#define     xch_fwhm                0.0     // FWHM of log_10(exchange) distribution
#define     xch_corr                0.0     // Correlation of exchange time to tau (rho_x in Kevin's paper; fast exchanges fast and slow exchanges slow)

#define     exch_type               2       // Exchange Method: (1) Uniform Dynamic Exchange (2) Kevin's Correlated Exchange

///////////////////////////////////////////////////////////////////////////////////////////////

void save_params(float seed) {

    FILE* param_file;
    char    param_name[80];

    sprintf(param_name, "params.txt");
    param_file = fopen(param_name, "w");

    fprintf(param_file, "Seed: %f\n", seed);

    fprintf(param_file, "Initial Position Selection: %d\n", init_pos);
    if (init_pos == 0){
        fprintf(param_file, "Features: %d\n\n", feats);
    }
    if (init_pos == 1) {
        fprintf(param_file, "Separation: %d\n\n", separation);
    }
   
    fprintf(param_file, "Frames: %d\n", frames);
    fprintf(param_file, "TBF: %f\n", tbf);
    fprintf(param_file, "NA: %f\n", aperture_detect);
    fprintf(param_file, "Anis. Ratio: %f\n", ratio);
    fprintf(param_file, "Point-Spread-Function FWHM: %d\n", PSF_fwhm);
    fprintf(param_file, "Pixel Size (nm): %d\n\n", pixel_size);

    fprintf(param_file, "Photons-Per-Frame: %f\n", photons_per_frame);
    fprintf(param_file, "Photomultiplier Gain: %f\n", photomultiplier_gain);
    fprintf(param_file, "Bkg Constant: %f\n", bkg_constant);
    fprintf(param_file, "Bkg Noise Amplitude: %f\n", bkg_amp);
    fprintf(param_file, "Add Excitation Contribution: %d\n\n", addExcitation);

    fprintf(param_file, "D med: %f\n", D_med);
    fprintf(param_file, "D FWHM: %f\n", D_fwhm);
    fprintf(param_file, "D corr: %f\n\n", D_corr);

    if (exch_type == 1) {
        fprintf(param_file, "D Exchange med: %d\n", Dxch_med);
    }
        
    else if(exch_type == 2){
        fprintf(param_file, "D Exchange med: %f\n", Dxch_med);
    }
        

    fprintf(param_file, "D Exchange FWHM: %f\n", Dxch_fwhm);
    fprintf(param_file, "D Exchange corr: %f\n\n", Dxch_corr);

    fprintf(param_file, "Tau med: %f\n", tau_med);
    fprintf(param_file, "Tau FWHM: %f\n", tau_fwhm);
    fprintf(param_file, "Tau corr: %f\n\n", tau_corr);

    fprintf(param_file, "Exchange med: %f\n", xch_med);
    fprintf(param_file, "Exchange FWHM: %f\n", xch_fwhm);
    fprintf(param_file, "Exchange corr: %f\n\n", xch_corr);

    fprintf(param_file, "Initial XY-Positions Method: %d\n", init_pos);
    fprintf(param_file, "Exchange Type: %d\n\n", exch_type);

    fclose(param_file);

    return;

}

void save_outputs(int frame,
    FILE* lc_file, FILE* rc_file, FILE* D_file, FILE* coord_file, FILE* steps_file, FILE* tau_file, FILE* ang_file, FILE* angsteps_file, FILE* totangstep_file, 
    char* lcname, char* rcname, char* Dname, char* coordname, char* stepsname, char* tauname, char* angname, char* angstepsname, char* totangstepname, 
    float* lc, float* rc, float* Darr, float* pos, float* steps, float* tauarr, float* ang, float* angsteps, float* totangstep, int numFeats) {

    // Save LC and RC frames

    sprintf(lcname, "./lc_frames/lc_frame_%d.txt", frame + 100001);
    sprintf(rcname, "./rc_frames/rc_frame_%d.txt", frame + 100001);

    lc_file = fopen(lcname, "w");
    rc_file = fopen(rcname, "w");

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            fprintf(lc_file, "%*.*f, ", DEC + 1, DEC, lc[i * WIDTH + j]);
            fprintf(rc_file, "%*.*f, ", DEC + 1, DEC, rc[i * WIDTH + j]);
        }
        fprintf(lc_file, "\n");
        fprintf(rc_file, "\n");
    }

    fclose(lc_file);
    fclose(rc_file);

    // Save all other outputs

    if (frame == 0) {

        sprintf(tauname, "./rot_data/all_tau.csv");
        tau_file = fopen(tauname, "w");
        sprintf(angname, "./rot_data/all_angles.csv");
        ang_file = fopen(angname, "w");
        sprintf(angstepsname, "./rot_data/all_angsteps.csv");
        angsteps_file = fopen(angstepsname, "w");
        sprintf(totangstepname, "./rot_data/all_total_anglesteps.csv");
        totangstep_file = fopen(totangstepname, "w");

        sprintf(Dname, "./trans_data/all_D.csv");
        D_file = fopen(Dname, "w");
        sprintf(coordname, "./trans_data/all_coords.csv");
        coord_file = fopen(coordname, "w");
        sprintf(stepsname, "./trans_data/all_steps.csv");
        steps_file = fopen(stepsname, "w");

        for (int i = 0; i < numFeats; i++) {

            if (i < numFeats - 1) {

                fprintf(tau_file, "%*.*f, ", DEC + 1, DEC, tauarr[i]);
                fprintf(ang_file, "%*.*f, %*.*f, ", DEC + 1, DEC, ang[i * 2 + 0], DEC + 1, DEC, ang[i * 2 + 1]);
                fprintf(angsteps_file, "%*.*f, %*.*f, ", DEC + 1, DEC, angsteps[i * 2 + 0], DEC + 1, DEC, angsteps[i * 2 + 1]);
                fprintf(totangstep_file, "%*.*f, ", DEC + 1, DEC, totangstep[i]);

                fprintf(D_file, "%*.*f, ", DEC + 1, DEC, Darr[i]);
                fprintf(coord_file, "%*.*f, %*.*f, ", DEC + 1, DEC, pos[i * 2 + 0], DEC + 1, DEC, pos[i * 2 + 1]);
                fprintf(steps_file, "%*.*f, %*.*f, ", DEC + 1, DEC, steps[i * 2 + 0], DEC + 1, DEC, steps[i * 2 + 1]);

            }
            else {

                fprintf(tau_file, "%*.*f\n", DEC + 1, DEC, tauarr[i]);
                fprintf(ang_file, "%*.*f, %*.*f\n", DEC + 1, DEC, ang[i * 2 + 0], DEC + 1, DEC, ang[i * 2 + 1]);
                fprintf(angsteps_file, "%*.*f, %*.*f\n", DEC + 1, DEC, angsteps[i * 2 + 0], DEC + 1, DEC, angsteps[i * 2 + 1]);
                fprintf(totangstep_file, "%*.*f\n", DEC + 1, DEC, totangstep[i]);

                fprintf(D_file, "%*.*f\n", DEC + 1, DEC, Darr[i]);
                fprintf(coord_file, "%*.*f, %*.*f\n", DEC + 1, DEC, pos[i * 2 + 0], DEC + 1, DEC, pos[i * 2 + 1]);
                fprintf(steps_file, "%*.*f, %*.*f\n", DEC + 1, DEC, steps[i * 2 + 0], DEC + 1, DEC, steps[i * 2 + 1]);

            }

        }

        fclose(tau_file);
        fclose(ang_file);
        fclose(angsteps_file);
        fclose(totangstep_file);

        fclose(D_file);
        fclose(coord_file);
        fclose(steps_file);

    }
    else {

        tau_file = fopen(tauname, "a");
        ang_file = fopen(angname, "a");
        angsteps_file = fopen(angstepsname, "a");
        totangstep_file = fopen(totangstepname, "a");

        D_file = fopen(Dname, "a");
        coord_file = fopen(coordname, "a");
        steps_file = fopen(stepsname, "a");

        for (int i = 0; i < numFeats; i++) {

            if (i < numFeats - 1) {

                fprintf(tau_file, "%*.*f, ", DEC + 1, DEC, tauarr[i]);
                fprintf(ang_file, "%*.*f, %*.*f, ", DEC + 1, DEC, ang[i * 2 + 0], DEC + 1, DEC, ang[i * 2 + 1]);
                fprintf(angsteps_file, "%*.*f, %*.*f, ", DEC + 1, DEC, angsteps[i * 2 + 0], DEC + 1, DEC, angsteps[i * 2 + 1]);
                fprintf(totangstep_file, "%*.*f, ", DEC + 1, DEC, totangstep[i]);

                fprintf(D_file, "%*.*f, ", DEC + 1, DEC, Darr[i]);
                fprintf(coord_file, "%*.*f, %*.*f, ", DEC + 1, DEC, pos[i * 2 + 0], DEC + 1, DEC, pos[i * 2 + 1]);
                fprintf(steps_file, "%*.*f, %*.*f, ", DEC + 1, DEC, steps[i * 2 + 0], DEC + 1, DEC, steps[i * 2 + 1]);

            }
            else {

                fprintf(tau_file, "%*.*f\n", DEC + 1, DEC, tauarr[i]);
                fprintf(ang_file, "%*.*f, %*.*f\n", DEC + 1, DEC, ang[i * 2 + 0], DEC + 1, DEC, ang[i * 2 + 1]);
                fprintf(angsteps_file, "%*.*f, %*.*f\n", DEC + 1, DEC, angsteps[i * 2 + 0], DEC + 1, DEC, angsteps[i * 2 + 1]);
                fprintf(totangstep_file, "%*.*f\n", DEC + 1, DEC, totangstep[i]);

                fprintf(D_file, "%*.*f\n", DEC + 1, DEC, Darr[i]);
                fprintf(coord_file, "%*.*f, %*.*f\n", DEC + 1, DEC, pos[i * 2 + 0], DEC + 1, DEC, pos[i * 2 + 1]);
                fprintf(steps_file, "%*.*f, %*.*f\n", DEC + 1, DEC, steps[i * 2 + 0], DEC + 1, DEC, steps[i * 2 + 1]);

            }

        }

        fclose(tau_file);
        fclose(ang_file);
        fclose(angsteps_file);
        fclose(totangstep_file);

        fclose(D_file);
        fclose(coord_file);
        fclose(steps_file);

    }

    return;

}

float sample_gaussian() {

    /////////////////////////////////////////////////////////////////////////////////
    ///// This function returns a random value sampled from a Gaussian distribution
    ///// with zero mean and unit variance
    /////////////////////////////////////////////////////////////////////////////////

    static int	index;
    static float	value[2];
    float	s, t, u, v;

    if (index == 0) {
        while (1) {
            u = (float)rand() / ((float)RAND_MAX / 2.0) - 1.0;
            v = (float)rand() / ((float)RAND_MAX / 2.0) - 1.0;
            s = u * u + v * v;
            if (s < 1.0)
                break;
        }

        t = sqrt(-2.0 * log(s) / s);
        value[0] = u * t;
        value[1] = v * t;
    }

    index = (index + 1) % 2;
    return value[index];

}

void local_boxmuller_transform(float A, float B, float* bm_return) {

    bm_return[0] = sqrt(-2 * log(A)) * cos(2 * PI * B);
    bm_return[1] = sqrt(-2 * log(A)) * sin(2 * PI * B);

    return;

}

float local_vector_Magnitude(float* vect) {

    float mag;

    mag = pow(vect[0], 2) + pow(vect[1], 2) + pow(vect[2], 2);


    return mag;

}

__global__ void reset_channels(float* lc, float* rc) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    lc[index] = 0.0f;
    rc[index] = 0.0f;

}

__device__ float gaussian1D(float var, double mean) {

    /////////////////////////////////////////////////////////////////////////////////
    ///// This function generates a Gaussian PSF to apply for to a feature relative to the feature center
    /////////////////////////////////////////////////////////////////////////////////

    float sigma = PSF_fwhm / C / pixel_size;

    return (1.0 / (sigma * sqrt(2 * PI))) * exp(-(pow(var - mean, 2) / (2 * pow(sigma, 2))));

}

__device__ double pixelPSF(double x_lLim, double x_uLim, double x_mean, double y_lLim, double y_uLim, double y_mean) {

    //double l_lim = -10.0, u_lim = 10.0; // l_lim: lower limit, u_lim: upper limit of integration
    int n = 100; // number of intervals
    double dx = (x_uLim - x_lLim) / n, dy = (y_uLim - y_lLim) / n; // step size; this is assuming that dx and dy stepsizes should be the same (true for symmetric Gaussian PSF)

    //double x_mean = 0.0, y_mean = 0.0; // this will need to change for the actual code and will be the sub-pixel localization position
    double xInteg = 0.0, yInteg = 0.0;

    for (int i = 0; i < n; i++) {

        xInteg = xInteg + (gaussian1D(x_lLim + i * dx, x_mean) + gaussian1D(x_lLim + (i + 1) * dx, x_mean)) * dx / 2.0;
        yInteg = yInteg + (gaussian1D(y_lLim + i * dy, y_mean) + gaussian1D(y_lLim + (i + 1) * dy, y_mean)) * dy / 2.0;

    }

    return xInteg * yInteg;

}

__global__ void apply_PSF(float* lc, float* rc, float* pos, float* ang) {

    /////////////////////////////////////////////////////////////////////////////////
    ///// This function applies a PSF with specified FWHM to each feature in the output movie frame
    /////////////////////////////////////////////////////////////////////////////////

    int index = threadIdx.x;

    float IPA = ang[(index * 2 + 0)];
    float OPA = ang[(index * 2 + 1)];
   
    float y_coord = pos[(index * 2 + 0)];
    float x_coord = pos[(index * 2 + 1)];

    float refractivity = 1.0f;
    float cos_alpha_detect, a_de, b_de, c_de;
    float aperture_ex = 0.001;
    float cos_alpha_ex, a_ex, b_ex;
    float factorLC, factorRC, exComp;
    int flr_x, flr_y, ceil_x, ceil_y, x_pixel, y_pixel;

    // Apply Fourkas

    cos_alpha_detect = cos(asin(aperture_detect / refractivity));

    a_de = (1.0 / 6.0) - (1.0 / 4.0) * cos_alpha_detect + (1.0 / 12.0) * pow(cos_alpha_detect, 3);
    b_de = (1.0 / 8.0) * cos_alpha_detect - (1.0 / 8.0) * pow(cos_alpha_detect, 3);
    c_de = (7.0 / 48.0) - (1.0 / 16.0) * cos_alpha_detect - (1.0 / 16.0) * pow(cos_alpha_detect, 2) - (1.0 / 48.0) *
        pow(cos_alpha_detect, 3);

    // Geometry

    factorLC = a_de + b_de * pow(sin(OPA), 2) + c_de * pow(sin(OPA), 2) * cos(2 * IPA);
    factorRC = a_de + b_de * pow(sin(OPA), 2) - c_de * pow(sin(OPA), 2) * cos(2 * IPA);

    // Excitation

    if (addExcitation == 1) {
        
        cos_alpha_ex = cos(asin(aperture_ex / refractivity));
        a_ex = (1.0 / 6.0) - (1.0 / 4.0) * cos_alpha_ex + (1.0 / 12.0) * pow(cos_alpha_ex, 3);
        b_ex = (1.0 / 8.0) * cos_alpha_ex - (1.0 / 8.0) * pow(cos_alpha_ex, 3);
        
        exComp = a_ex + b_ex * pow(sin(OPA), 2);

        factorLC *= exComp;
        factorRC *= exComp;
    }

    // Mean Intensity

    factorLC *= photons_per_frame * photomultiplier_gain;
    factorRC *= photons_per_frame * photomultiplier_gain;

    flr_x = floor(x_coord);
    flr_y = floor(y_coord);
    ceil_x = ceil(x_coord);
    ceil_y = ceil(y_coord);

    if (fabs(flr_x - x_coord) < fabs(ceil_x - x_coord))
        x_pixel = flr_x;
    else
        x_pixel = ceil_x;

    if (fabs(flr_y - y_coord) < fabs(ceil_y - y_coord))
        y_pixel = flr_y;
    else
        y_pixel = ceil_y;

    for (int i = y_pixel - 3; i < y_pixel + 4; i++) {
        for (int j = x_pixel - 3; j < x_pixel + 4; j++) {

            //lc[i * WIDTH + j] += PSF_gaussian(x_coord, j) * PSF_gaussian(y_coord, i) * factorLC;
            //rc[i * WIDTH + j] += PSF_gaussian(x_coord, j) * PSF_gaussian(y_coord, i) * factorRC;

            lc[i * WIDTH + j] += pixelPSF(float(j) - 0.5, float(j) + 0.5, x_coord, float(i) - 0.5, float(i) + 0.5, y_coord) * factorLC;
            rc[i * WIDTH + j] += pixelPSF(float(j) - 0.5, float(j) + 0.5, x_coord, float(i) - 0.5, float(i) + 0.5, y_coord) * factorRC;

        }
    }

    return;

}

__device__ void boxmuller_transform(float A, float B, float *bm_return) {

    bm_return[0] = sqrt(-2 * log(A)) * cos(2 * PI * B);
    bm_return[1] = sqrt(-2 * log(A)) * sin(2 * PI * B);

    return;
}

__device__ void crossProduct(float* crossP, float* vect_A, float* vect_B) {

    crossP[0] = vect_A[1] * vect_B[2] - vect_A[2] * vect_B[1];
    crossP[1] = vect_A[2] * vect_B[0] - vect_A[0] * vect_B[2];
    crossP[2] = vect_A[0] * vect_B[1] - vect_A[1] * vect_B[0];

    return;

}

__device__ float dotProduct(float* vect_A, float* vect_B) {

    float product = 0;

    for (int i = 0; i < 3; i++) {

        product = product + vect_A[i] * vect_B[i];

    }

    return product;

}

__device__ void rodriguesRotation(float* rot_vect, float* init_vect, float* crossP, float angle, float* fin_vect) {

    float out1[3], out2[3], out3[3];
    // float dotProd;

    // Compute First Term
    for (int i = 0; i < 3; i++) {
        out1[i] = cos(angle) * init_vect[i];
    }

    // Compute Second Term

    crossProduct(crossP, rot_vect, init_vect);

    for (int i = 0; i < 3; i++) {
        out2[i] = sin(angle) * crossP[i];
    }

    // Compute Third Term

    for (int i = 0; i < 3; i++) {
        out3[i] = (1 - cos(angle)) * dotProduct(rot_vect, init_vect) * rot_vect[i];
    }

    for (int i = 0; i < 3; i++) {
        fin_vect[i] = out1[i] + out2[i] + out3[i];
    }

    return;

}

__device__ float sample_rayleigh(float rand) {

    /////////////////////////////////////////////////////////////////////////////////
    ///// This function returns a random value sampled from a Rayleigh distribution
    ///// with unit width
    /////////////////////////////////////////////////////////////////////////////////

    return sqrt(-2.0 * logf(rand));
}

 __device__ void translate(float* pos, float *steps, int index, float jump, float a, float b) {

     float rand_vect[2];
     float disp_vect[2];

     // Generate a Random 2D Unit Vector and Scale it by Jump Size

     boxmuller_transform(a, b, rand_vect);

     for (int i = 0; i < 2; i++) {
         disp_vect[i] = rand_vect[i] / sqrt(pow(rand_vect[0], 2) + pow(rand_vect[1], 2)) * jump;
     }
     
     // Translate the Initial Position by this Vector
        // It's arbitrary but lets say disp_vect[0] is y-coord and disp_vect[1] is x-coord

     for (int i = 0; i < 2; i++) {
         pos[index * 2 + i] += disp_vect[i];
         steps[index * 2 + i] = disp_vect[i];
     }

     return;

    }

 __device__ void rotate(float* ang, float* angsteps, float* totangstep, int index, float angle, float a, float b, float c, float d, float axisAng) {

     float cartDipole[3], normDipole[3];
     float orthVect[3], normOrth[3];
     float tempVect[3], normTemp[3];
     float axisVect[3], normAxis[3];
     float finDipole[3], normFinDipole[3];

     float crossP[3];

     // Convert the Angular Dipole to Cartesian Coordinates
        // Remember that ang[0] is IPA and ang[1] is OPA

     cartDipole[0] = sin(ang[index * 2 + 1]) * cos(ang[index * 2 + 0]);
     cartDipole[1] = sin(ang[index * 2 + 1]) * sin(ang[index * 2 + 0]);
     cartDipole[2] = cos(ang[index * 2 + 1]);

        // Renormalize to Eliminate Potential Rounding Errors

     for (int i = 0; i < 3; i++) {
         normDipole[i] = cartDipole[i] / sqrt(pow(cartDipole[0], 2) + pow(cartDipole[1], 2) + pow(cartDipole[2], 2));
     }

     // Generate the Random Axis about which Rotation will Take Place

        // Generate an arbitrary, simple orthogonal vector to the Dipole Vector

     if (normDipole[2] != 1) {
         orthVect[0] = normDipole[1];
         orthVect[1] = -normDipole[0];
         orthVect[2] = 0;
     }

     else {
         orthVect[0] = normDipole[2];
         orthVect[1] = 0;
         orthVect[2] = -normDipole[0];
     }

            // Renormalize to Eliminate Potential Rounding Errors

     for (int i = 0; i < 3; i++) {
         normOrth[i] = orthVect[i] / sqrt(pow(orthVect[0], 2) + pow(orthVect[1], 2) + pow(orthVect[2], 2));
     }

        // Take the cross product of the dipole vector and the arbitrary orthogonal vector

     crossProduct(tempVect, normDipole, normOrth);

            // Renormalize to Eliminate Potential Rounding Errors

     for (int i = 0; i < 3; i++) {
         normTemp[i] = tempVect[i] / sqrt(pow(tempVect[0], 2) + pow(tempVect[1], 2) + pow(tempVect[2], 2));
     }

        // Take a Linear Combination of the arbitrary orthogonal vector and the cross product vector with randomly, uniformly distributed angle [0, 2pi]

     for (int i = 0; i < 3; i++) {
         axisVect[i] = cos(axisAng) * normOrth[i] + sin(axisAng) * normTemp[i];
     }

            // Renormalize to Eliminate Potential Rounding Errors
     
     for (int i = 0; i < 3; i++) {
         normAxis[i] = axisVect[i] / sqrt(pow(axisVect[0], 2) + pow(axisVect[1], 2) + pow(axisVect[2], 2));
     }

     // Rotate by Applying Rodrigues' Rotation Formula

     rodriguesRotation(normAxis, normDipole, crossP, angle, finDipole);

        // Renormalize to Eliminate Potential Rounding Errors

     for (int i = 0; i < 3; i++) {
         normFinDipole[i] = finDipole[i] / sqrt(pow(finDipole[0], 2) + pow(finDipole[1], 2) + pow(finDipole[2], 2));
     }

     // Convert Cartesian Dipole to Polar Again

     ang[2 * index + 0] = atan(normFinDipole[1] / normFinDipole[0]); // IPA
     ang[2 * index + 1] = acos(normFinDipole[2]); // OPA

    // angsteps[2 * index + 0] = 

     float product = 0;
     for (int i = 0; i < 3; i++) {
         product = product + cartDipole[i] * normFinDipole[i];
     }
     
     totangstep[index] = acos(product);

     return;

 }

 __global__ void random_walk(float* pos, float* steps, float* ang, float* angsteps, float* totangstep, float* Drand, float* Dexchrand, float* prevDarr, float* Darr, float* taurand, float* tauexchrand, float* prevtauarr, float* tauarr, float D_sigma, float tau_sigma, float frame, int* Dxch, int* tauxch, float Dxch_sigma, float tauxch_sigma) {

     int index = threadIdx.x;
     curandState state;
     curand_init(clock64(), index, 0, &state);

     float jump, tau, ang_jump;
     float rayleighrand_trans, transBM_a, transBM_b, rayleighrand_rot, rotBM_a, rotBM_b, rotBM_c, rotBM_d, rotAxisAng;
     float tau_rand, xch_rand;

     // Decrement Exchange Counter

     Dxch[index]--;
     tauxch[index]--;

     // Sample Drand and Taurand on a Gaussian if Respective Exchange Cts = 0

     if (Dxch[index] == 0) {
         
         prevDarr[index] = Darr[index];
         Dexchrand[index] = curand_normal(&state);
         Drand[index] = curand_normal(&state);
     }

     if (tauxch[index] <= 0) {
         
         prevtauarr[index] = tauarr[index];
         tauexchrand[index] = curand_normal(&state);
         taurand[index] = curand_normal(&state);
     }

     // Generate Random Parameters

        // For Translation
     rayleighrand_trans = curand_uniform(&state);
     transBM_a = curand_uniform(&state);
     transBM_b = curand_uniform(&state);

     // For Rotation
     rayleighrand_rot = curand_uniform(&state);
     rotBM_a = curand_uniform(&state);
     rotBM_b = curand_uniform(&state);
     rotBM_c = curand_uniform(&state);
     rotBM_d = curand_uniform(&state);
     rotAxisAng = curand_uniform(&state) * 2 * PI;

     // Determine Stepsizes for both Modes

     // For Translation
            
         // *** NO CORRELATED EXCHANGE IS IMPLEMENTED HERE YET ***

     // nm^2 / s

     jump = powf(10.0, D_sigma * Drand[index] + D_med);
     Darr[index] = jump;

     // pixel^2 / frame
     jump = sqrtf(2 * jump * tbf) / pixel_size;

     // For Rotation

     if (exch_type == 1) {
         tau = powf(10.0, tau_sigma * taurand[index] + tau_med);
     }

     else if (exch_type == 2) {
         tau_rand = tau_corr * prevtauarr[index] + sqrt(1 - tau_corr * tau_corr) * taurand[index];
         tau = powf(10.0, tau_sigma * tau_rand + tau_med);
     }

     // (s) timescale
     
     tauarr[index] = tau;

     // radians / frame
     ang_jump = sqrtf(1.0 * tbf / (3.0 * tau));

     // // If Respective Exchange Cts = 0 then Reset Exchange Counter

     if (Dxch[index] == 0) {

         // Uncorrelated, Uniform Exchange Dynamic Heterogeneity
         if (exch_type == 1) {
             Dxch[index] = Dxch_med;
         }

         // NOT WORRYING ABOUT TRANSLATION FOR THE TIME BEING; THAT CAN BE IMPLEMENTED AT A LATER DATE 8/29/2024

         // IMPLEMENT A PARAMETER WHICH LETS YOU SELECT THE TYPE OF EXCHANGE YOU WANT TO USE

         //Dxch[index] = (int)powf(10, log10f(Dxch_med) - 0.5 * log10f(Darr[index]) / powf(10, D_med));   // Han's method of exchange time and correlation --> Dxch_med is in terms of frames akin to Talha's approach
         //Dxch[index] = (int)(powf(10.0, Dxch_sigma * curand_normal(&state) + Dxch_med) / tbf); // Unlike tauxch this can't be viewed with reference to D_med b/c of units --> best to view relative to tau_med actually I think so exchange happens roughly the same timescale for now
         //Dxch[index] = (int)(Dxch_med - (D_med * 10));  // Here Dxch_med is the median number of frames not some related timescale (following Talha/Nicki Sim Paper definition)

     }

     if (tauxch[index] <= 0) {

         // Uncorrelated, Uniform Exchange Dynamic Heterogeneity
         if (exch_type == 1) {
             
             tauxch[index] = xch_med;
         }
                  
         // Kevin's approach: Correlate exchange times with timescale experienced AND Correlate dynamic timescales with prior dynamic timescales
         else if (exch_type == 2) {
             
             // medianXch = log10(xch_med * tbf);
             
             xch_rand = xch_corr * tau_rand + sqrt(1 - xch_corr * xch_corr) * tauexchrand[index];
             tauxch[index] = (int)(pow(10.0, tauxch_sigma * xch_rand + xch_med) + 0.5);
         }


         // IMPLEMENT A PARAMETER WHICH LETS YOU SELECT THE TYPE OF EXCHANGE YOU WANT TO USE

         //tauxch[index] = (int)powf(10, log10f(xch_med) - 0.5 * log10f(((1 / (6 * tauarr[index])) / (1 / (6 * powf(10, tau_med)))))); // Han's method of exchange time and correlation --> tauxch_med is in terms of frames akin to Talha's approach

         //tauxch[index] = (int)(powf(10.0, tauxch_sigma * curand_normal(&state) + xch_med) / tbf); // Approach like Kevin was doing

     }

     // Translate 

     translate(pos, steps, index, jump * sample_rayleigh(rayleighrand_trans), transBM_a, transBM_b);

     // Rotate

     rotate(ang, angsteps, totangstep, index, ang_jump * sample_rayleigh(rayleighrand_rot), rotBM_a, rotBM_b, rotBM_c, rotBM_d, rotAxisAng);

     return;

 }

 __global__ void add_noise(float* lc, float* rc) {

     /////////////////////////////////////////////////////////////////////////////////
     ///// This function adds a static bkg intensity and random bkg noise to the output .tifs
     /////////////////////////////////////////////////////////////////////////////////

     // I would like to make this a bit more sophisticated in the future though exactly how I am unsure of

     int index = threadIdx.x + blockIdx.x * blockDim.x;

     curandState state;
     curand_init(clock64(), index, 0, &state);

     // Add background floor

     lc[index] += bkg_constant * photons_per_frame * photomultiplier_gain;
     rc[index] += bkg_constant * photons_per_frame * photomultiplier_gain;

     // Add background noise

     lc[index] += curand_normal(&state) * photons_per_frame * photomultiplier_gain * bkg_amp;
     rc[index] += curand_normal(&state) * photons_per_frame * photomultiplier_gain * bkg_amp;

     // Add detector noise

     //lc[index] += curand_normal(&state) * sqrtf(lc[index] * 2.0);
     //rc[index] += curand_normal(&state) * sqrtf(rc[index] * 2.0);

     return;

 }

 int main() {

     // Save Parameters File

     float seed = time(0);
     srand(seed);
     save_params(seed);

     float D_sigma = (float)D_fwhm / C;
     float tau_sigma = (float)tau_fwhm / C;

     float Dxch_sigma = (float)Dxch_fwhm / C;
     float xch_sigma = (float)xch_fwhm / C;

     // Initialize Folders and Files for Saving Results

     FILE* lc_file, * rc_file, * D_file, * coord_file, * steps_file, * tau_file, * ang_file, * angsteps_file, * totangstep_file;
     char    lcname[80], rcname[80], Dname[80], coordname[80], stepsname[80], tauname[80], angname[80], angstepsname[80], totangstepname[80];
     char    rotdatadir[300], transdatadir[300], lcdir[300], rcdir[300];

     _getcwd(rotdatadir, sizeof(rotdatadir));
     _getcwd(transdatadir, sizeof(transdatadir));
     _getcwd(lcdir, sizeof(lcdir));
     _getcwd(rcdir, sizeof(rcdir));
     strcat(rotdatadir, "/rot_data");
     strcat(transdatadir, "/trans_data");
     strcat(lcdir, "/lc_frames");
     strcat(rcdir, "/rc_frames");
     _mkdir(rotdatadir);
     _mkdir(transdatadir);
     _mkdir(lcdir);
     _mkdir(rcdir);

     // Rotation and Translation Data Arrays

     if (init_pos == 0) {

        // Initialize and Allocate Space for Host (CPU) Copies of Data Arrays

            // LC and RC Movie File Arrays

         float* lc = (float*)malloc(HEIGHT * WIDTH * sizeof(float));
         float* rc = (float*)malloc(HEIGHT * WIDTH * sizeof(float));

         for (int i = 0; i < HEIGHT; i++) {
             for (int j = 0; j < WIDTH; j++) {
                 lc[i * WIDTH + j] = 0.0f;
                 rc[i * WIDTH + j] = 0.0f;
             }
         }

         float* ang = (float*)malloc(2 * feats * sizeof(float));
         float* angsteps = (float*)malloc(2 * feats * sizeof(float));
         float* totangstep = (float*)malloc(feats * sizeof(float));
         float* taurand = (float*)malloc(feats * sizeof(float));
         float* tauarr = (float*)malloc(feats * sizeof(float));

         float* pos = (float*)malloc(2 * feats * sizeof(float));
         float* steps = (float*)malloc(2 * feats * sizeof(float));
         float* Drand = (float*)malloc(feats * sizeof(float));
         float* Darr = (float*)malloc(feats * sizeof(float));

         float* prevtauarr = (float*)malloc(feats * sizeof(float));
         float* prevDarr = (float*)malloc(feats * sizeof(float));

         float* tauexchrand = (float*)malloc(feats * sizeof(float));
         float* Dexchrand = (float*)malloc(feats * sizeof(float));

         float dipole[3];
         float bm_return[2];
         float a, b, c, d;

         float* dipole_save = (float*)malloc(3 * feats * sizeof(float));
         float* bmParams = (float*)malloc(4 * feats * sizeof(float));

         for (int i = 0; i < feats; i++) {

             // Reset Dipole to Zero

             for (int j = 0; j < 3; j++) {
                 dipole[j] = 0.0f;
             }

             // For Rotation: Set Tau and Step Arrays to Zero; Set Angle Array to Random Initial Vector

             if (i > 0) {
                 prevtauarr[i] = tauarr[i];
             }

             tauarr[i] = 0.0f;
             taurand[i] = 0.0f;

             angsteps[i * 2 + 0] = 0.0f;
             angsteps[i * 2 + 1] = 0.0f;
             totangstep[i] = 0.0f;

             // For Generating Random Initial Vector

                 // First Generate Uniform Random Parameters

             a = (float)rand() / (float)RAND_MAX;
             b = (float)rand() / (float)RAND_MAX;
             c = (float)rand() / (float)RAND_MAX;
             d = (float)rand() / (float)RAND_MAX;

             // Convert to Normally Random Vector Components using Box-Muller Transform

             local_boxmuller_transform(a, b, bm_return);
             dipole[0] = bm_return[0];
             dipole[1] = bm_return[1];

             while (dipole[0] == 0 || a == 0) {

                 a = (float)rand() / (float)RAND_MAX;
                 b = (float)rand() / (float)RAND_MAX;

                 local_boxmuller_transform(a, b, bm_return);
                 dipole[0] = bm_return[0];
                 dipole[1] = bm_return[1];

             }

             local_boxmuller_transform(c, d, bm_return);
             dipole[2] = bm_return[0];

             while (dipole[2] == 0 || c == 0) {

                 c = (float)rand() / (float)RAND_MAX;
                 d = (float)rand() / (float)RAND_MAX;

                 local_boxmuller_transform(c, d, bm_return);
                 dipole[2] = bm_return[0];

             }

             bmParams[i * 4 + 0] = a;
             bmParams[i * 4 + 1] = b;
             bmParams[i * 4 + 2] = c;
             bmParams[i * 4 + 3] = d;

             for (int j = 0; j < 3; j++) {
                 dipole_save[i * 3 + j] = dipole[j];
             }

             // Convert from a Cartestian Vector to Polar Coordinates

             ang[i * 2 + 0] = atan(dipole[1] / dipole[0]);                              // IPA
             ang[i * 2 + 1] = acos(dipole[2] / sqrt(local_vector_Magnitude(dipole)));   // OPA

             // For Translation: Set D and Step Arrays to Zero, Set Initial Position within FOV and Sufficiently Away from the Edge; 

             if (i > 0) {
                 prevDarr[i] = Darr[i];
             }

             Darr[i] = 0.0f;
             Drand[i] = 0.0f;

             steps[i * 2 + 0] = 0.0f;
             steps[i * 2 + 1] = 0.0f;

             pos[i * 2 + 0] = ((float)rand() / (float)RAND_MAX) * HEIGHT;       // Random initial Y-position
             pos[i * 2 + 1] = ((float)rand() / (float)RAND_MAX) * WIDTH;    // Random initial X-position

             while (pos[i * 2 + 0] <= 5 || pos[i * 2 + 0] > HEIGHT - 5 || pos[i * 2 + 1] <= 5 || pos[i * 2 + 1] > WIDTH - 5) {

                pos[i * 2 + 0] = ((float)rand() / (float)RAND_MAX) * HEIGHT;    // Random initial Y-pos
                pos[i * 2 + 1] = ((float)rand() / (float)RAND_MAX) * WIDTH;     // Random initial X-pos

             }
         }

         // For Exchange Times

         int* Dxch = (int*)malloc(feats * sizeof(int));
         int* tauxch = (int*)malloc(feats * sizeof(int));

         for (int i = 0; i < feats; i++) {
             Dxch[i] = 1;
             tauxch[i] = 1;
         }

         // Define and Allocate Space for Device (GPU) Copies of Data Arrays

         int* d_Dxch, * d_tauxch;
         float* d_lc, * d_rc;
         float* d_pos, * d_steps, * d_Drand, * d_Darr;
         float* d_ang, * d_angsteps, * d_totangstep, * d_taurand, * d_tauarr;
         float* d_Dexchrand, * d_tauexchrand, * d_prevDarr, * d_prevtauarr; 

         cudaMalloc((void**)&d_lc, HEIGHT * WIDTH * sizeof(float));
         cudaMalloc((void**)&d_rc, HEIGHT * WIDTH * sizeof(float));

         cudaMalloc((void**)&d_pos, 2 * feats * sizeof(float));
         cudaMalloc((void**)&d_steps, 2 * feats * sizeof(float));
         cudaMalloc((void**)&d_Drand, feats * sizeof(float));
         cudaMalloc((void**)&d_Darr, feats * sizeof(float));
         cudaMalloc((void**)&d_Dxch, feats * sizeof(int));

         cudaMalloc((void**)&d_ang, 2 * feats * sizeof(float));
         cudaMalloc((void**)&d_angsteps, 2 * feats * sizeof(float));
         cudaMalloc((void**)&d_totangstep, feats * sizeof(float));
         cudaMalloc((void**)&d_taurand, feats * sizeof(float));
         cudaMalloc((void**)&d_tauarr, feats * sizeof(float));
         cudaMalloc((void**)&d_tauxch, feats * sizeof(int));

         cudaMalloc((void**)&d_Dexchrand, feats * sizeof(float));
         cudaMalloc((void**)&d_tauexchrand, feats * sizeof(float));
         cudaMalloc((void**)&d_prevDarr, feats * sizeof(float));
         cudaMalloc((void**)&d_prevtauarr, feats * sizeof(float));

         // Run Analysis

         for (int framect = 0; framect < frames; framect++) {

             printf("\r ON FRAME: %d/%d", framect + 1, frames);
             fflush(stdout);

             if (framect == 0) {

                 // Apply Camera Parameters and PSF

                 cudaMemcpy(d_lc, lc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_rc, rc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice);

                 cudaMemcpy(d_pos, pos, feats * 2 * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_ang, ang, feats * 2 * sizeof(float), cudaMemcpyHostToDevice);

                 apply_PSF << <1, feats >> > (d_lc, d_rc, d_pos, d_ang);

                 cudaMemcpy(lc, d_lc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(rc, d_rc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

                 cudaMemcpy(ang, d_ang, feats * 2 * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(pos, d_pos, feats * 2 * sizeof(float), cudaMemcpyDeviceToHost);

                 // Add Noise

                 cudaMemcpy(d_lc, lc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_rc, rc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice);

                 add_noise << <HEIGHT, WIDTH >> > (d_lc, d_rc);

                 cudaMemcpy(lc, d_lc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(rc, d_rc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

                 // Save Outputs

                 save_outputs(framect, lc_file, rc_file, D_file, coord_file, steps_file, tau_file, ang_file, angsteps_file, totangstep_file,
                     lcname, rcname, Dname, coordname, stepsname, tauname, angname, angstepsname, totangstepname,
                     lc, rc, Darr, pos, steps, tauarr, ang, angsteps, totangstep, feats);

             } else {

                 // Reset LC and RC intensity values to 0.0 for all pixels

                 cudaMemcpy(d_lc, lc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_rc, rc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice);

                 reset_channels << <HEIGHT, WIDTH >> > (d_lc, d_rc);

                 cudaMemcpy(lc, d_lc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(rc, d_rc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

                 // Random Walk for XY-position and Dipole Orientation

                 cudaMemcpy(d_pos, pos, feats * 2 * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_steps, steps, feats * 2 * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_Drand, Drand, feats * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_Darr, Darr, feats * sizeof(float), cudaMemcpyHostToDevice);

                 cudaMemcpy(d_ang, ang, feats * 2 * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_angsteps, angsteps, feats * 2 * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_totangstep, totangstep, feats * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_taurand, taurand, feats * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_tauarr, tauarr, feats * sizeof(float), cudaMemcpyHostToDevice);

                 cudaMemcpy(d_prevtauarr, prevtauarr, feats * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_prevDarr, prevDarr, feats * sizeof(float), cudaMemcpyHostToDevice);

                 cudaMemcpy(d_Dxch, Dxch, feats * sizeof(int), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_tauxch, tauxch, feats * sizeof(int), cudaMemcpyHostToDevice);

                 cudaMemcpy(d_Dexchrand, Dexchrand, feats * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_tauexchrand, tauexchrand, feats * sizeof(float), cudaMemcpyHostToDevice);

                 random_walk << <1, feats >> > (d_pos, d_steps, d_ang, d_angsteps, d_totangstep, d_Drand, d_Dexchrand, d_prevDarr, d_Darr, d_taurand, d_tauexchrand, d_prevtauarr, d_tauarr, D_sigma, tau_sigma, framect, d_Dxch, d_tauxch, Dxch_sigma, xch_sigma);

                 cudaMemcpy(pos, d_pos, feats * 2 * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(steps, d_steps, feats * 2 * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(Drand, d_Drand, feats * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(Darr, d_Darr, feats * sizeof(float), cudaMemcpyDeviceToHost);

                 cudaMemcpy(ang, d_ang, feats * 2 * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(angsteps, d_angsteps, feats * 2 * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(totangstep, d_totangstep, feats * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(taurand, d_taurand, feats * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(tauarr, d_tauarr, feats * sizeof(float), cudaMemcpyDeviceToHost);

                 cudaMemcpy(prevtauarr, d_prevtauarr, feats * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(prevDarr, d_prevDarr, feats * sizeof(float), cudaMemcpyDeviceToHost);

                 cudaMemcpy(Dxch, d_Dxch, feats * sizeof(int), cudaMemcpyDeviceToHost);
                 cudaMemcpy(tauxch, d_tauxch, feats * sizeof(int), cudaMemcpyDeviceToHost);

                 cudaMemcpy(Dexchrand, d_Dexchrand, feats * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(tauexchrand, d_tauexchrand, feats * sizeof(float), cudaMemcpyDeviceToHost);

                 // Apply Camera Parameters and PSF

                 cudaMemcpy(d_lc, lc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_rc, rc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice);

                 apply_PSF << <1, feats >> > (d_lc, d_rc, d_pos, d_ang);

                 cudaMemcpy(lc, d_lc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(rc, d_rc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

                 // Add Noise

                 cudaMemcpy(d_lc, lc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_rc, rc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice);

                 add_noise << <HEIGHT, WIDTH >> > (d_lc, d_rc);

                 cudaMemcpy(lc, d_lc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(rc, d_rc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

                 // Save Outputs

                 save_outputs(framect, lc_file, rc_file, D_file, coord_file, steps_file, tau_file, ang_file, angsteps_file, totangstep_file,
                     lcname, rcname, Dname, coordname, stepsname, tauname, angname, angstepsname, totangstepname,
                     lc, rc, Darr, pos, steps, tauarr, ang, angsteps, totangstep, feats);

             }
         }
     }

     else if (init_pos == 1) {

         // Initialize and Allocate Space for Host (CPU) Copies of Data Arrays

             // LC and RC Movie File Arrays

         float* lc = (float*)malloc(HEIGHT * WIDTH * sizeof(float));
         float* rc = (float*)malloc(HEIGHT * WIDTH * sizeof(float));

         for (int i = 0; i < HEIGHT; i++) {
             for (int j = 0; j < WIDTH; j++) {
                 lc[i * WIDTH + j] = 0.0f;
                 rc[i * WIDTH + j] = 0.0f;
             }
         }

         // Determine Number of Features to Simulate Given Separation Constraint

         int xFeatDim = 0;
         int yFeatDim = 0;

         for (int i = 1; i < WIDTH; i++) {
             if (i * separation > WIDTH - separation) {
                 break;
             }
             else {
                 xFeatDim++;
             }
         }

         for (int i = 1; i < HEIGHT; i++) {
             if (i * separation > HEIGHT - separation) {
                 break;
             }
             else {
                 yFeatDim++;
             }
         }

         int numFeats = xFeatDim * yFeatDim;

         // Initialize and Allocate Space for other Host (CPU) Copies of Data Arrays

         float* ang = (float*)malloc(2 * numFeats * sizeof(float));
         float* angsteps = (float*)malloc(2 * numFeats * sizeof(float));
         float* totangstep = (float*)malloc(numFeats * sizeof(float));
         float* taurand = (float*)malloc(numFeats * sizeof(float));
         float* tauarr = (float*)malloc(numFeats * sizeof(float));

         float* pos = (float*)malloc(2 * numFeats * sizeof(float));
         float* steps = (float*)malloc(2 * numFeats * sizeof(float));
         float* Drand = (float*)malloc(numFeats * sizeof(float));
         float* Darr = (float*)malloc(numFeats * sizeof(float));

         float* prevtauarr = (float*)malloc(feats * sizeof(float));
         float* prevDarr = (float*)malloc(feats * sizeof(float));
         
         float* tauexchrand = (float*)malloc(feats * sizeof(float));
         float* Dexchrand = (float*)malloc(feats * sizeof(float));

         float dipole[3];
         float bm_return[2];
         float a, b, c, d;

         float* dipole_save = (float*)malloc(3 * numFeats * sizeof(float));
         float* bmParams = (float*)malloc(4 * numFeats * sizeof(float));

         for (int i = 0; i < numFeats; i++) {

             // Reset Dipole to Zero

             for (int j = 0; j < 3; j++) {
                 dipole[j] = 0.0f;
             }

             // For Rotation: Set Tau and Step Arrays to Zero; Set Angle Array to Random Initial Vector

             tauarr[i] = 0.0f;
             taurand[i] = 0.0f;

             angsteps[i * 2 + 0] = 0.0f;
             angsteps[i * 2 + 1] = 0.0f;
             totangstep[i] = 0.0f;

             // For Generating Random Initial Vector

                 // First Generate Uniform Random Parameters

             a = (float)rand() / (float)RAND_MAX;
             b = (float)rand() / (float)RAND_MAX;
             c = (float)rand() / (float)RAND_MAX;
             d = (float)rand() / (float)RAND_MAX;

             // Convert to Normally Random Vector Components using Box-Muller Transform

             local_boxmuller_transform(a, b, bm_return);
             dipole[0] = bm_return[0];
             dipole[1] = bm_return[1];

             while (dipole[0] == 0 || a == 0) {

                 a = (float)rand() / (float)RAND_MAX;
                 b = (float)rand() / (float)RAND_MAX;

                 local_boxmuller_transform(a, b, bm_return);
                 dipole[0] = bm_return[0];
                 dipole[1] = bm_return[1];

             }

             local_boxmuller_transform(c, d, bm_return);
             dipole[2] = bm_return[0];

             while (dipole[2] == 0 || c == 0) {

                 c = (float)rand() / (float)RAND_MAX;
                 d = (float)rand() / (float)RAND_MAX;

                 local_boxmuller_transform(c, d, bm_return);
                 dipole[2] = bm_return[0];

             }

             bmParams[i * 4 + 0] = a;
             bmParams[i * 4 + 1] = b;
             bmParams[i * 4 + 2] = c;
             bmParams[i * 4 + 3] = d;

             for (int j = 0; j < 3; j++) {
                 dipole_save[i * 3 + j] = dipole[j];
             }

             // Convert from a Cartestian Vector to Polar Coordinates

             ang[i * 2 + 0] = atan(dipole[1] / dipole[0]);                              // IPA
             ang[i * 2 + 1] = acos(dipole[2] / sqrt(local_vector_Magnitude(dipole)));   // OPA

             // For Translation: Set D and Step Arrays to Zero, Set Initial Position within FOV and Sufficiently Away from the Edge; 

             Darr[i] = 0.0f;
             Drand[i] = 0.0f;

             steps[i * 2 + 0] = 0.0f;
             steps[i * 2 + 1] = 0.0f;

             // Determine Initial Positions for All Features 

             pos[i * 2 + 0] = ((float(i % xFeatDim)) + 1.0) * separation; // Y-Coordinate
             pos[i * 2 + 1] = ((float(i / xFeatDim)) + 1.0) * separation; // X-Coordinate

         }

         // For Exchange Times

         int* Dxch = (int*)malloc(numFeats * sizeof(int));
         int* tauxch = (int*)malloc(numFeats * sizeof(int));

         for (int i = 0; i < numFeats; i++) {
             Dxch[i] = 1;
             tauxch[i] = 1;
         }

         // Define and Allocate Space for Device (GPU) Copies of Data Arrays

         int* d_Dxch, * d_tauxch;
         float* d_lc, * d_rc;
         float* d_pos, * d_steps, * d_Drand, * d_Darr;
         float* d_ang, * d_angsteps, * d_totangstep, * d_taurand, * d_tauarr;
         float* d_Dexchrand, * d_tauexchrand, * d_prevDarr, * d_prevtauarr;

         cudaMalloc((void**)&d_lc, HEIGHT* WIDTH * sizeof(float));
         cudaMalloc((void**)&d_rc, HEIGHT* WIDTH * sizeof(float));

         cudaMalloc((void**)&d_pos, 2 * numFeats * sizeof(float));
         cudaMalloc((void**)&d_steps, 2 * numFeats * sizeof(float));
         cudaMalloc((void**)&d_Drand, numFeats * sizeof(float));
         cudaMalloc((void**)&d_Darr, numFeats * sizeof(float));
         cudaMalloc((void**)&d_Dxch, numFeats * sizeof(int));

         cudaMalloc((void**)&d_ang, 2 * numFeats * sizeof(float));
         cudaMalloc((void**)&d_angsteps, 2 * numFeats * sizeof(float));
         cudaMalloc((void**)&d_totangstep, numFeats * sizeof(float));
         cudaMalloc((void**)&d_taurand, numFeats * sizeof(float));
         cudaMalloc((void**)&d_tauarr, numFeats * sizeof(float));
         cudaMalloc((void**)&d_tauxch, numFeats * sizeof(int));

         cudaMalloc((void**)&d_Dexchrand, feats * sizeof(float));
         cudaMalloc((void**)&d_tauexchrand, feats * sizeof(float));
         cudaMalloc((void**)&d_prevDarr, feats * sizeof(float));
         cudaMalloc((void**)&d_prevtauarr, feats * sizeof(float));

         // Run Analysis

         for (int framect = 0; framect < frames; framect++) {

             printf("\r ON FRAME: %d/%d", framect + 1, frames);
             fflush(stdout);

             if (framect == 0) {

                 // Apply Camera Parameters and PSF

                 cudaMemcpy(d_lc, lc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_rc, rc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice);

                 cudaMemcpy(d_pos, pos, numFeats * 2 * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_ang, ang, numFeats * 2 * sizeof(float), cudaMemcpyHostToDevice);

                 apply_PSF << <1, numFeats >> > (d_lc, d_rc, d_pos, d_ang);

                 cudaMemcpy(lc, d_lc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(rc, d_rc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

                 cudaMemcpy(ang, d_ang, numFeats * 2 * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(pos, d_pos, numFeats * 2 * sizeof(float), cudaMemcpyDeviceToHost);

                 // Add Noise

                 cudaMemcpy(d_lc, lc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_rc, rc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice);

                 add_noise << <HEIGHT, WIDTH >> > (d_lc, d_rc);

                 cudaMemcpy(lc, d_lc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(rc, d_rc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

                 // Save Outputs

                 save_outputs(framect, lc_file, rc_file, D_file, coord_file, steps_file, tau_file, ang_file, angsteps_file, totangstep_file,
                     lcname, rcname, Dname, coordname, stepsname, tauname, angname, angstepsname, totangstepname,
                     lc, rc, Darr, pos, steps, tauarr, ang, angsteps, totangstep, numFeats);

             } else {

                 // Reset LC and RC intensity values to 0.0 for all pixels

                 cudaMemcpy(d_lc, lc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_rc, rc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice);

                 reset_channels << <HEIGHT, WIDTH >> > (d_lc, d_rc);

                 cudaMemcpy(lc, d_lc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(rc, d_rc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

                 // Random Walk for XY-position and Dipole Orientation

                 cudaMemcpy(d_pos, pos, numFeats * 2 * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_steps, steps, numFeats * 2 * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_Drand, Drand, numFeats * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_Darr, Darr, numFeats * sizeof(float), cudaMemcpyHostToDevice);

                 cudaMemcpy(d_ang, ang, numFeats * 2 * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_angsteps, angsteps, numFeats * 2 * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_totangstep, totangstep, numFeats * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_taurand, taurand, numFeats * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_tauarr, tauarr, numFeats * sizeof(float), cudaMemcpyHostToDevice);

                 cudaMemcpy(d_prevtauarr, prevtauarr, numFeats * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_prevDarr, prevDarr, numFeats * sizeof(float), cudaMemcpyHostToDevice);

                 cudaMemcpy(d_Dxch, Dxch, numFeats * sizeof(int), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_tauxch, tauxch, numFeats * sizeof(int), cudaMemcpyHostToDevice);

                 cudaMemcpy(d_Dexchrand, Dexchrand, numFeats * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_tauexchrand, tauexchrand, numFeats * sizeof(float), cudaMemcpyHostToDevice);

                 random_walk << <1, numFeats >> > (d_pos, d_steps, d_ang, d_angsteps, d_totangstep, d_Drand, d_Dexchrand, d_prevDarr, d_Darr, d_taurand, d_tauexchrand, d_prevtauarr, d_tauarr, D_sigma, tau_sigma, framect, d_Dxch, d_tauxch, Dxch_sigma, xch_sigma);

                 cudaMemcpy(pos, d_pos, numFeats * 2 * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(steps, d_steps, numFeats * 2 * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(Drand, d_Drand, numFeats * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(Darr, d_Darr, numFeats * sizeof(float), cudaMemcpyDeviceToHost);

                 cudaMemcpy(ang, d_ang, numFeats * 2 * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(angsteps, d_angsteps, numFeats * 2 * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(totangstep, d_totangstep, numFeats * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(taurand, d_taurand, numFeats * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(tauarr, d_tauarr, numFeats * sizeof(float), cudaMemcpyDeviceToHost);

                 cudaMemcpy(prevtauarr, d_prevtauarr, numFeats * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(prevDarr, d_prevDarr, numFeats * sizeof(float), cudaMemcpyDeviceToHost);

                 cudaMemcpy(Dxch, d_Dxch, numFeats * sizeof(int), cudaMemcpyDeviceToHost);
                 cudaMemcpy(tauxch, d_tauxch, numFeats * sizeof(int), cudaMemcpyDeviceToHost);

                 cudaMemcpy(Dexchrand, d_Dexchrand, numFeats * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(tauexchrand, d_tauexchrand, numFeats * sizeof(float), cudaMemcpyDeviceToHost);

                 // Apply Camera Parameters and PSF

                 cudaMemcpy(d_lc, lc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_rc, rc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice);

                 apply_PSF << <1, numFeats >> > (d_lc, d_rc, d_pos, d_ang);

                 cudaMemcpy(lc, d_lc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(rc, d_rc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

                 // Add Noise

                 cudaMemcpy(d_lc, lc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
                 cudaMemcpy(d_rc, rc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice);

                 add_noise << <HEIGHT, WIDTH >> > (d_lc, d_rc);

                 cudaMemcpy(lc, d_lc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);
                 cudaMemcpy(rc, d_rc, HEIGHT * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

                 // Save Outputs

                 save_outputs(framect, lc_file, rc_file, D_file, coord_file, steps_file, tau_file, ang_file, angsteps_file, totangstep_file,
                     lcname, rcname, Dname, coordname, stepsname, tauname, angname, angstepsname, totangstepname,
                     lc, rc, Darr, pos, steps, tauarr, ang, angsteps, totangstep, numFeats);

             }
         }
     }

     return 0;

 }