/** Copyright 2020 RWTH Aachen University. All rights reserved.
 *
 *  Licensed under the RWTH ASR License (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.hltpr.rwth-aachen.de/rwth-asr/rwth-asr-license.html
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
#include <Core/Assertions.hh>
#include <Core/Utility.hh>

#include "FastHartleyTransform.hh"

namespace Signal {

FastHartleyTransform::FastHartleyTransform(const u32 length,
                                           const f32 sampleRate) {
    setLength(length);
    setSampleRate(sampleRate);
}

void FastHartleyTransform::setBitReserve(u32 length) {
    bitReverse_.resize(length);

    for (u32 k1 = 1, k2 = 0; k1 < length; k1++) {
        for (u32 k = length >> 1; (!((k2 ^= k) & k)); k >>= 1)
            ;
        bitReverse_[k1] = k2;
    }
}

#define GOOD_TRIG

#ifndef GOOD_TRIG
#define FAST_TRIG
#endif

#if defined(GOOD_TRIG)
#define TRIG_VARS u32 t_lam = 0;
#define TRIG_INIT(k, c, s)         \
    {                              \
        u32 i;                     \
        for (i = 1; i <= k; i++) { \
            coswrk[i] = costab[i]; \
            sinwrk[i] = sintab[i]; \
        }                          \
        t_lam = 0;                 \
        c     = 1;                 \
        s     = 0;                 \
    }
#define TRIG_NEXT(k, c, s)                                       \
    {                                                            \
        u32 i, j;                                                \
        t_lam++;                                                 \
        for (i = 0; !((1 << i) & t_lam); i++)                    \
            ;                                                    \
        i = k - i;                                               \
        s = sinwrk[i];                                           \
        c = coswrk[i];                                           \
        if (i > 1) {                                             \
            for (j = k - i + 2; (1 << j) & t_lam; j++)           \
                ;                                                \
            j         = k - j;                                   \
            sinwrk[i] = halsec[i] * (sinwrk[i - 1] + sinwrk[j]); \
            coswrk[i] = halsec[i] * (coswrk[i - 1] + coswrk[j]); \
        }                                                        \
    }
#endif

#if defined(FAST_TRIG)
#define TRIG_VARS double t_c, t_s;
#define TRIG_INIT(k, c, s) \
    {                      \
        t_c = costab[k];   \
        t_s = sintab[k];   \
        c   = 1;           \
        s   = 0;           \
    }
#define TRIG_NEXT(k, c, s)            \
    {                                 \
        double t = c;                 \
        c        = t * t_c - s * t_s; \
        s        = t * t_s + s * t_c; \
    }
#endif

static double halsec[20] = {
        0,
        0,
        .54119610014619698439972320536638942006107206337801,
        .50979557910415916894193980398784391368261849190893,
        .50241928618815570551167011928012092247859337193963,
        .50060299823519630134550410676638239611758632599591,
        .50015063602065098821477101271097658495974913010340,
        .50003765191554772296778139077905492847503165398345,
        .50000941253588775676512870469186533538523133757983,
        .50000235310628608051401267171204408939326297376426,
        .50000058827484117879868526730916804925780637276181,
        .50000014706860214875463798283871198206179118093251,
        .50000003676714377807315864400643020315103490883972,
        .50000000919178552207366560348853455333939112569380,
        .50000000229794635411562887767906868558991922348920,
        .50000000057448658687873302235147272458812263401372};
static double costab[20] = {
        .00000000000000000000000000000000000000000000000000,
        .70710678118654752440084436210484903928483593768847,
        .92387953251128675612818318939678828682241662586364,
        .98078528040323044912618223613423903697393373089333,
        .99518472667219688624483695310947992157547486872985,
        .99879545620517239271477160475910069444320361470461,
        .99969881869620422011576564966617219685006108125772,
        .99992470183914454092164649119638322435060646880221,
        .99998117528260114265699043772856771617391725094433,
        .99999529380957617151158012570011989955298763362218,
        .99999882345170190992902571017152601904826792288976,
        .99999970586288221916022821773876567711626389934930,
        .99999992646571785114473148070738785694820115568892,
        .99999998161642929380834691540290971450507605124278,
        .99999999540410731289097193313960614895889430318945,
        .99999999885102682756267330779455410840053741619428};
static double sintab[20] = {
        1.0000000000000000000000000000000000000000000000000,
        .70710678118654752440084436210484903928483593768846,
        .38268343236508977172845998403039886676134456248561,
        .19509032201612826784828486847702224092769161775195,
        .09801714032956060199419556388864184586113667316749,
        .04906767432741801425495497694268265831474536302574,
        .02454122852291228803173452945928292506546611923944,
        .01227153828571992607940826195100321214037231959176,
        .00613588464915447535964023459037258091705788631738,
        .00306795676296597627014536549091984251894461021344,
        .00153398018628476561230369715026407907995486457522,
        .00076699031874270452693856835794857664314091945205,
        .00038349518757139558907246168118138126339502603495,
        .00019174759731070330743990956198900093346887403385,
        .00009587379909597734587051721097647635118706561284,
        .00004793689960306688454900399049465887274686668768};

#define SQRT2_2 0.70710678118654752440084436210484
#define SQRT2 2 * 0.70710678118654752440084436210484

void FastHartleyTransform::hartleyTransform(std::vector<Data>& fz) const {
    // *very* fast hartley transform
    f32 *fi, *fn, *gi;
    u32  i, k, k1, k2, k3, k4, kx;
#ifdef GOOD_TRIG
    double coswrk[20], sinwrk[20];
#endif
    TRIG_VARS;

    for (k1 = 1; k1 < length(); k1++) {
        k2 = bitReverse_[k1];
        if (k1 > k2) {
            f32 a  = fz[k1];
            fz[k1] = fz[k2];
            fz[k2] = a;
        }
    }

    for (k = 0; u32(1 << k) < length(); k++)
        ;
    k &= 1;

    if (k == 0) {
        for (fi = &fz[0], fn = &fz[length()]; fi < fn; fi += 4) {
            double f0, f1, f2, f3;
            f1    = fi[0] - fi[1];
            f0    = fi[0] + fi[1];
            f3    = fi[2] - fi[3];
            f2    = fi[2] + fi[3];
            fi[2] = (f0 - f2);
            fi[0] = (f0 + f2);
            fi[3] = (f1 - f3);
            fi[1] = (f1 + f3);
        }
    }
    else {
        for (fi = &fz[0], fn = &fz[length()], gi = fi + 1; fi < fn; fi += 8,
            gi += 8) {
            double s1, c1, s2, c2, s3, c3, s4, c4, g0, f0, f1, g1, f2, g2, f3, g3;
            c1    = fi[0] - gi[0];
            s1    = fi[0] + gi[0];
            c2    = fi[2] - gi[2];
            s2    = fi[2] + gi[2];
            c3    = fi[4] - gi[4];
            s3    = fi[4] + gi[4];
            c4    = fi[6] - gi[6];
            s4    = fi[6] + gi[6];
            f1    = (s1 - s2);
            f0    = (s1 + s2);
            g1    = (c1 - c2);
            g0    = (c1 + c2);
            f3    = (s3 - s4);
            f2    = (s3 + s4);
            g3    = SQRT2 * c4;
            g2    = SQRT2 * c3;
            fi[4] = f0 - f2;
            fi[0] = f0 + f2;
            fi[6] = f1 - f3;
            fi[2] = f1 + f3;
            gi[4] = g0 - g2;
            gi[0] = g0 + g2;
            gi[6] = g1 - g3;
            gi[2] = g1 + g3;
        }
    }
    if (length() < 16)
        return;

    do {
        double s1, c1;
        k += 2;
        k1 = 1 << k;
        k2 = k1 << 1;
        k4 = k2 << 1;
        k3 = k2 + k1;
        kx = k1 >> 1;
        fi = &fz[0];
        gi = fi + kx;
        fn = &fz[length()];
        do {
            double g0, f0, f1, g1, f2, g2, f3, g3;
            f1     = fi[0] - fi[k1];
            f0     = fi[0] + fi[k1];
            f3     = fi[k2] - fi[k3];
            f2     = fi[k2] + fi[k3];
            fi[k2] = f0 - f2;
            fi[0]  = f0 + f2;
            fi[k3] = f1 - f3;
            fi[k1] = f1 + f3;
            g1     = gi[0] - gi[k1];
            g0     = gi[0] + gi[k1];
            g3     = SQRT2 * gi[k3];
            g2     = SQRT2 * gi[k2];
            gi[k2] = g0 - g2;
            gi[0]  = g0 + g2;
            gi[k3] = g1 - g3;
            gi[k1] = g1 + g3;
            gi += k4;
            fi += k4;
        } while (fi < fn);

        TRIG_INIT(k, c1, s1);
        for (i = 1; i < kx; i++) {
            double c2, s2;
            TRIG_NEXT(k, c1, s1);
            c2 = c1 * c1 - s1 * s1;
            s2 = 2 * (c1 * s1);
            fn = &fz[length()];
            fi = &fz[i];
            gi = &fz[k1 - i];
            do {
                double a, b, g0, f0, f1, g1, f2, g2, f3, g3;
                b      = s2 * fi[k3] - c2 * gi[k3];
                a      = c2 * fi[k3] + s2 * gi[k3];
                f3     = fi[k2] - a;
                f2     = fi[k2] + a;
                g3     = gi[k2] - b;
                g2     = gi[k2] + b;
                b      = s2 * fi[k1] - c2 * gi[k1];
                a      = c2 * fi[k1] + s2 * gi[k1];
                f1     = fi[0] - a;
                f0     = fi[0] + a;
                g1     = gi[0] - b;
                g0     = gi[0] + b;
                b      = s1 * f2 - c1 * g3;
                a      = c1 * f2 + s1 * g3;
                fi[k2] = f0 - a;
                fi[0]  = f0 + a;
                gi[k3] = g1 - b;
                gi[k1] = g1 + b;
                b      = c1 * g2 - s1 * f3;
                a      = s1 * g2 + c1 * f3;
                gi[k2] = g0 - a;
                gi[0]  = g0 + a;
                fi[k3] = f1 - b;
                fi[k1] = f1 + b;
                gi += k4;
                fi += k4;
            } while (fi < fn);
        }
    } while (k4 < length());
}

void FastHartleyTransform::zeroPadding(std::vector<Data>& data) const {
    ensure(data.size() <= length());

    data.resize(length(), 0);
}

void FastHartleyTransform::transform(std::vector<Data>& data) const {
    zeroPadding(data);

    hartleyTransform(data);

    if (sampleRate_ != 1) {
        std::transform(data.begin(), data.end(), data.begin(),
                       std::bind2nd(std::multiplies<Data>(), 1 / (Data)sampleRate_));
    }
}

void FastHartleyTransform::inverseTransform(std::vector<Data>& data) const {
    zeroPadding(data);

    hartleyTransform(data);

    std::transform(data.begin(), data.end(), data.begin(),
                   std::bind2nd(std::multiplies<Data>(), (Data)sampleRate_ / (Data)length()));
}

void hartleyToFourier(const std::vector<f32>& hartley, std::vector<f32>& fourier) {
    if (hartley.empty())
        return;

    u32 N = hartley.size();

    fourier.resize((N / 2 + 1) * 2);  // N even: N + 2; N odd: N + 1

    fourier[0] = hartley[0];
    fourier[1] = 0;

    for (u32 n = 1; n <= N / 2; ++n) {
        fourier[2 * n]     = (hartley[n] + hartley[N - n]) / 2;
        fourier[2 * n + 1] = (hartley[n] - hartley[N - n]) / 2;
    }
}

void hartleyToFourierAmplitude(const std::vector<f32>& hartley, std::vector<f32>& amplitude) {
    if (hartley.empty())
        return;

    u32 N = hartley.size();

    if (amplitude.size() < N / 2 + 1)
        amplitude.resize(N / 2 + 1);

    amplitude[0] = Core::abs(hartley[0]);

    f32 real2, imaginary2;

    for (u32 n = 1; n <= N / 2; ++n) {
        real2      = hartley[n] + hartley[N - n];
        imaginary2 = hartley[n] - hartley[N - n];

        amplitude[n] = sqrt(real2 * real2 + imaginary2 * imaginary2) / 2;
    }

    amplitude.resize(N / 2 + 1);
}

void hartleyToFourierPhase(const std::vector<f32>& hartley, std::vector<f32>& phase) {
    if (hartley.empty())
        return;

    u32 N = hartley.size();

    if (phase.size() < N / 2 + 1)
        phase.resize(N / 2 + 1);

    phase[0] = atan2(0, hartley[0]);

    for (u32 n = 1; n <= N / 2; ++n)
        phase[n] = atan2(hartley[n] - hartley[N - n], hartley[n] + hartley[N - n]);

    phase.resize(N / 2 + 1);
}

void fourierToHartley(const std::vector<f32>& fourier, std::vector<f32>& hartley) {
    if (fourier.size() < 2)
        return;

    ensure(*(fourier.begin() + 1) == 0);

    u32 N = fourier.size() - (fourier.back() == 0 ? 2 : 1);

    hartley.resize(N);

    hartley[0] = fourier[0];

    for (u32 n = 1; n <= N / 2; ++n) {
        hartley[n]     = fourier[2 * n] + fourier[2 * n + 1];
        hartley[N - n] = fourier[2 * n] - fourier[2 * n + 1];
    }
}
}  // namespace Signal
