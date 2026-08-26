# Packet Compute Layout Fixed-Clock Attribution

## Batch 16

| metric | aggregate | bitmap interleaved | bitmap warp-row |
|:--|--:|--:|--:|
| time (us) | 1200.608 | 1222.144 | 1196.192 |
| warp instructions | 176,653,966 | 193,632,912 | 191,596,562 |
| CBU instructions | 4,502,163 | 4,933,475 | 5,220,584 |
| LSU instructions | 22,853,281 | 22,778,932 | 22,397,573 |
| global-load sectors | 29,184,580 | 31,449,833 | 28,726,861 |
| shared-load conflicts | 6,643,420 | 8,379,469 | 6,797,157 |
| shared-store conflicts | 4,391,374 | 4,339,187 | 4,346,601 |
| barrier stall (%) | 21.340 | 17.500 | 17.770 |
| long scoreboard (%) | 29.310 | 30.300 | 29.480 |
| short scoreboard (%) | 1.610 | 1.740 | 1.800 |
| registers/thread | 64.000 | 64.000 | 64.000 |

Warp-row/interleaved: time=0.979x; instructions=0.989x; load sectors=0.913x; shared-load conflicts=0.811x; CBU instructions=1.058x; LSU instructions=0.983x.

## Batch 32

| metric | aggregate | bitmap interleaved | bitmap warp-row |
|:--|--:|--:|--:|
| time (us) | 2854.816 | 2855.904 | 2735.648 |
| warp instructions | 360,721,463 | 394,356,052 | 389,723,288 |
| CBU instructions | 12,150,191 | 12,881,296 | 12,918,909 |
| LSU instructions | 47,057,462 | 46,860,921 | 45,854,376 |
| global-load sectors | 58,853,272 | 63,247,771 | 57,659,427 |
| shared-load conflicts | 13,255,298 | 16,292,858 | 13,166,798 |
| shared-store conflicts | 8,706,811 | 8,640,275 | 8,649,276 |
| barrier stall (%) | 32.360 | 29.330 | 28.770 |
| long scoreboard (%) | 26.520 | 26.130 | 24.530 |
| short scoreboard (%) | 1.190 | 1.360 | 1.410 |
| registers/thread | 64.000 | 64.000 | 64.000 |

Warp-row/interleaved: time=0.958x; instructions=0.988x; load sectors=0.912x; shared-load conflicts=0.808x; CBU instructions=1.003x; LSU instructions=0.979x.

## Batch 64

| metric | aggregate | bitmap interleaved | bitmap warp-row |
|:--|--:|--:|--:|
| time (us) | 5670.912 | 5599.168 | 5723.296 |
| warp instructions | 721,256,630 | 788,082,378 | 781,184,671 |
| CBU instructions | 24,692,154 | 25,421,274 | 26,712,354 |
| LSU instructions | 94,283,984 | 93,559,691 | 92,106,581 |
| global-load sectors | 117,584,298 | 126,474,566 | 115,505,888 |
| shared-load conflicts | 26,417,354 | 32,720,203 | 26,521,407 |
| shared-store conflicts | 17,403,809 | 17,292,535 | 17,282,995 |
| barrier stall (%) | 32.280 | 28.950 | 30.060 |
| long scoreboard (%) | 26.300 | 27.010 | 25.890 |
| short scoreboard (%) | 1.180 | 1.370 | 1.350 |
| registers/thread | 64.000 | 64.000 | 64.000 |

Warp-row/interleaved: time=1.022x; instructions=0.991x; load sectors=0.913x; shared-load conflicts=0.811x; CBU instructions=1.051x; LSU instructions=0.984x.

CUDA-event timing remains the ranking authority. Promote warp-row only if its predicted conflict/sector reduction is visible and is not offset by instruction or scoreboard growth.
