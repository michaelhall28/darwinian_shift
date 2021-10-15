#include <iostream>
#include "TMath.h"
#include "TError.h"
#include "TString.h"
#include "TF1.h"
#include "TF2.h"
#include "Math/Integrator.h"

//following includes for illustration only
#include "TRandom3.h"
using namespace std;

// Homogeneity test for weighted data samples - Kolmogorov-Smirnov, Cramér-von Mises, Anderson-Darling test
// if a sample is unweighted pass an array of ones
Double_t *WHomogeneityTest(Int_t na, const Double_t *a, const Double_t *wa, Int_t nb, const Double_t *b, const Double_t *wb, Option_t *option)
{
   TString opt = option;
   opt.ToUpper();
   Double_t *prob = new Double_t[3];
   Double_t *stat = new Double_t[3];
   for (Int_t i = 0; i < 3; i++) {
      prob[i] = -1;
      stat[i] = 0;
   }
// Require at least two points in each graph
   if (!a || !b || !wa || !wb || na <= 2 || nb <= 2) {
      Error("HomogeneityTest", "Sets must have more than 2 points");
      return prob;
   }
// Constants needed
   Int_t ia = 0;
   Int_t ib = 0;
   Double_t eventsa = 0;
   Double_t eventsb = 0;
   Double_t sq_eventsa = 0;
   Double_t sq_eventsb = 0;

// Calculating effective entries size
   for (Int_t i = 0; i < na; i++) {
      eventsa += wa[i];
      sq_eventsa += pow(wa[i], 2.0);
   }
   for (Int_t j = 0; j < nb; j++) {
      eventsb += wb[j];
      sq_eventsb += pow(wb[j], 2.0);
   }
   Double_t effna = pow(eventsa, 2.0) / sq_eventsa;
   Double_t effnb = pow(eventsb, 2.0) / sq_eventsb;
   Double_t effn = effna + effnb;

// Auxiliary variables
   Double_t x;
   Double_t sumwa = 0;
   Double_t sumwb = 0;
   Double_t rdmax = 0;
   Double_t FA = 0;
   Double_t FB = 0;
   Double_t H;
   Bool_t enda = false;
   Bool_t endb = false;

// Main loop over point sets
   while ((!enda) && (!endb)) {
      if (enda) {
         x = b[ib];
      }
      else if (endb) {
         x = a[ia];
      }
      else {
         x = TMath::Min(a[ia], b[ib]);
      }
      while (a[ia] == x && ia < na) {
         sumwa += wa[ia];
         ia++;
      }
      while (b[ib] == x && ib < nb) {
         sumwb += wb[ib];
         ib++;
      }
      FA += sumwa / eventsa;
      FB += sumwb / eventsb;
      H = (effna * FA + effnb * FB) / effn;
      rdmax = TMath::Max(rdmax, TMath::Abs(FA - FB));
      stat[1] += TMath::Power(FA - FB, 2.0) * (sumwa / eventsa * effna + sumwb / eventsb * effnb) / effn;
      if ((H > 0) && (H < 1)) {
         stat[2] += TMath::Power(FA - FB, 2.0) / ((1 - H) * H) * (sumwa / eventsa * effna + sumwb / eventsb * effnb) / effn;
      }

      // reseting sumwa, sumwb
      sumwa = 0;
      sumwb = 0;

      // set last point to infinity
      if (ia == na) {
         enda = true;
      }
      if (ib == nb) {
         endb = true;
      }
   }

// Computing KS test's p-value
   Double_t z = rdmax * TMath::Sqrt(effna * effnb / (effna + effnb));
   prob[0] = TMath::KolmogorovProb(z);
   stat[0] = rdmax;

// Computing CvM test's p-value
   stat[1] = effna * effnb / effn * stat[1];
   Double_t sum = 0;
   Double_t num;

// GSL prints errors even though it works perfectly
   gErrorIgnoreLevel = kBreak;
   if (stat[1] > 2.7) {
      sum = 1;
   }
   else {
      TF2 *f1 = new TF2("f1", "ROOT::Math::cyl_bessel_k(0.25,pow(4.0*x+1.0,2.0)/(16.0*y))", 0, TMath::Infinity());
      for (Int_t j = 0; j < 21; j++) {
         num = f1->Eval(j, stat[1], 0);
         sum += 1 / (TMath::Pi() * sqrt(stat[1])) * pow(-1, j) * TMath::Gamma(0.5) / (TMath::Gamma(j + 1.0) * TMath::Gamma(0.5 - j)) * sqrt(4 * j + 1.0) * exp(-pow(4 * j + 1.0, 2.0) / (16 * stat[1])) * num;
      }
	  delete f1;
   }
   prob[1] = 1 - sum;

// Computing AD test's p-value
   sum = 0;
   stat[2] = effna * effnb / effn * stat[2];
   if (stat[2] > 13.2) {
      sum = 1;
   }
   else {
      TF1 *f2 = new TF1("f2", "exp([0]/(8*(x^2+1))-(TMath::Pi()^2*x^2*(4*[1]+1)^2)/(8*[0]))", 0, TMath::Infinity());
      f2->SetParameter(0, stat[2]);
      for (Int_t j = 0; j < 21; j++) {
         f2->SetParameter(1, (double)j);
         num = f2->Integral(0, TMath::Infinity(), 0);
         sum += sqrt(2 * TMath::Pi()) / stat[2] * TMath::Gamma(0.5) / (TMath::Gamma(j + 1.0) * TMath::Gamma(0.5 - j)) * (4 * j + 1.0) * exp(-pow((4 * j + 1.0) * TMath::Pi(), 2.0) / (8 * stat[2])) * num;
      }
	  delete f2;
   }
   prob[2] = 1 - sum;
   gErrorIgnoreLevel = kUnset;

   // debug printout
   if (opt.Contains("D")) {
      printf(" Kolmogorov Probability = %g, Max Dist = %g\n", prob[0], stat[0]);
      printf(" Cramér-von Mises p-value = %g, Test statistic = %g\n", prob[1], stat[1]);
      printf(" Anderson-Darling p-value = %g, Test statistic = %g\n", prob[2], stat[2]);
   }
   if (opt.Contains("M")) {
	  delete[] prob;
	  return stat;
   } else {
	  delete[] stat;
	  return prob;
   }
}

// illustration of tests
int homtests()
{
   TRandom3 myRandom(21535);
   Double_t *entriesA = new Double_t[100];
   Double_t *entriesB = new Double_t[200];
   Double_t *weightsA = new Double_t[100];
   Double_t *weightsB = new Double_t[200];
   for (Int_t i = 0; i < 100; i++) {
      entriesA[i] = myRandom.Gaus(0, 1);
      weightsA[i] = myRandom.Exp(10);
      entriesB[2 * i] = myRandom.Gaus(0, 1);
      weightsB[2 * i] = myRandom.Exp(4);
      entriesB[2 * i + 1] = myRandom.Gaus(0, 1);
      weightsB[2 * i + 1] = myRandom.Exp(3);
   }
   std::sort(&entriesA[0], &entriesA[100]);
   std::sort(&entriesB[0], &entriesB[200]);

   Double_t *pval = WHomogeneityTest(100, entriesA, weightsA, 200, entriesB, weightsB, "D");
   printf("\nKS pval = %g, CvM pval = %g, AD pval = %g\n", pval[0], pval[1], pval[2]);

   delete[] entriesA;
   delete[] entriesB;
   delete[] weightsA;
   delete[] weightsB;
   return 0;
}
