#define helperFunctions_cxx


#include <ROOT/RVec.hxx>

using VecF_t = const ROOT::RVec<float>&;
using VecI_t = const ROOT::RVec<int>&;
using VecB_t = const ROOT::VecOps::RVec<bool>;

#include "helperFunctions.h"




int C = 0.15; 

bool myfilter(float x) {
   return x > 5;
}

auto sum = [](int a, int b) {
        return a + b;
    };


// int num_baseline_lep(VecF_t& pt, VecF_t& eta, VecI_t& fllep, VecB_t passOR, VecB_t passLOOSE, VecB_t passMEDIUM, VecB_t passBL, VecF_t z0sinth){
//   int nbl = 0;
//   for(unsigned int i=0; i<fllep.size(); i++)
//     {
//       if(pt[i] < 9)continue;
//       if((fllep[i] == 1 && fabs(eta[i])>2.47) || ((fllep[i] == 2 && fabs(eta[i])>2.6)))continue;
//       if(!passOR[i])continue
//       if((fllep[i] == 1 && (!passLOOSE[i] || !passBL[i])) || (fllep[i] == 22 && !passMEDIUM[i]))continue;
//       if(fabs(z0sinth)<0.5)continue;
//       nbl += 1;
//     }
//   return nbl;
// }

Double_t getMetRel(VecF_t& pt, VecF_t& eta, VecF_t& phi, VecF_t& e, Float_t met_et, Float_t met_phi){
  TLorentzVector l;
  Double_t min_dphi_lep_met = 9999;
  TLorentzVector met;
  met.SetPtEtaPhiM(met_et, 0.0, met_phi, 0.0);
  for(unsigned int i=0; i<pt.size(); i++)
    {
      l.SetPtEtaPhiM(pt[i], eta[i], phi[i], e[i]);
      Double_t dphi = fabs(l.DeltaPhi(met));
      if(dphi < min_dphi_lep_met){
	min_dphi_lep_met = dphi;
      }
    }
  return (min_dphi_lep_met < M_PI/2.0) ? (met_et)*sin(min_dphi_lep_met) : (met_et);
 
}

std::pair <int,int> num_bl_sg_lep(VecF_t& pt, VecF_t& eta, VecI_t& fllep, VecB_t passOR, VecB_t passLOOSE, VecB_t passMEDIUM, VecB_t passBL, VecF_t z0sinth, VecB_t ISO, VecF_t d0sig, VecF_t passTIGHT){
  int nbl = 0;
  int nsg = 0;
  std::pair <int,int> nlep;
  for(unsigned int i=0; i<fllep.size(); i++)
    {
      if(pt[i] < 9)continue;
      if((fllep[i] == 1 && fabs(eta[i])>2.47) || ((fllep[i] == 2 && fabs(eta[i])>2.6)))continue;
      if(!passOR[i])continue;
      if((fllep[i] == 1 && (!passLOOSE[i] || !passBL[i])) || (fllep[i] == 2 && !passMEDIUM[i]))continue;
      if(fabs(z0sinth[i])>0.5)continue;
      
      nbl += 1;
      
      if((fllep[i] == 1 && !passTIGHT[i]))continue;
      if(!ISO[i])continue;
      if((fllep[i] == 1 && fabs(d0sig[i])>5) || ((fllep[i] == 2 && fabs(d0sig[i])>3)))continue;
      
      nsg += 1;
    }
  nlep = std::make_pair(nbl,nsg);
  return nlep;
}


std::pair <double,double> getLeptonsFromZ(VecI_t chlep, VecI_t& fllep, VecF_t& pt, VecF_t& eta, VecF_t& phi, VecF_t& e, Float_t met_et, Float_t met_phi){
  double diff = 10000000000.0;
  /**
  int Zlep1 = -99;
  int Zlep2 = -99;
  int Wlep1 = -999;
  */
  double Zmass = -1.0;
  double Wmass = -1.0;
  bool foundSFOS = false;
  std::pair <double,double> masses;
  for(unsigned int i=0; i<chlep.size(); i++)
    {
      for(unsigned int j=i+1; j<chlep.size(); j++)
	{
	  //Opposite-Sign
	  if(chlep[i]*chlep[j]<0)
	    {
	      //Same-Flavor
	      if(abs(fllep[i])==abs(fllep[j]))
		{
		  TLorentzVector p1;
		  p1.SetPtEtaPhiM(pt[i], eta[i], phi[i], e[i]);
		  TLorentzVector p2;
		  p2.SetPtEtaPhiM(pt[j], eta[j], phi[j], e[j]);
		  double mass = (p1+p2).M();
		  double massdiff = fabs(mass-91187.6);
		  if(massdiff<diff)
		    {
		      diff=massdiff;
		      Zmass=mass;
		      Zlep1 = i;
		      Zlep2 = j;
		      foundSFOS = true;
		    }
		}
	    }
	}

    }
  
  if(foundSFOS){
    TLorentzVector met;
    met.SetPtEtaPhiM(met_et, 0.0, met_phi, 0.0);
    
    if((Zlep1==0 && Zlep2==1) || (Zlep1==1 && Zlep2==0) ) Wlep1=2;
    else if((Zlep1==0 && Zlep2==2) || (Zlep1==2 && Zlep2==0) ) Wlep1=1;
    else if((Zlep1==1 && Zlep2==2) || (Zlep1==2 && Zlep2==1) ) Wlep1=0;
    
    TLorentzVector lepW;
    lepW.SetPtEtaPhiM(pt[Wlep1], eta[Wlep1], phi[Wlep1], e[Wlep1]);
    double wlepMetphi = lepW.DeltaPhi(met);
    Wmass = sqrt(2*lepW.Pt()*met.Pt()*(1-cos(wlepMetphi)));
  }
  masses = std::make_pair(Zmass,Wmass);
    
  return masses;
}

Float_t getLumiSF(Int_t randrnum){
    Float_t lumi15 = 3219.56;
    Float_t lumi16 = 32988.1;
    Float_t lumi17 = 44307.4;
    Float_t lumi18 = 58450.1;
    if(randrnum < 320000)return lumi15+lumi16;
    else if(randrnum > 320000 && randrnum < 348000)return lumi17;
    else if(randrnum > 348000)return lumi18;
    else{std::cout<<"ERROR \t RandomRunnumber "<<randrnum<<" has no period attached"<<std::endl;}
    return 1.0;
}

double getSF(VecF_t& sf){
  const auto size = sf.size();
  double scalef = 1.0;
  for (size_t i=0; i < size; ++i) {
    scalef *= sf[i];
  }
  return scalef;
}

bool isOS(const ROOT::VecOps::RVec<int>& chlep) {
  if(chlep[0]*chlep[1] < 0)return kTRUE;
  return kFALSE;
}

// bool isTriggerMatched(const ROOT::VecOps::RVec<int>& isTM) {
//   std::vector<int> tm_vec; 
//   const auto size = isTM.size();
//    for (size_t i=0; i < size; ++i) {
//      if(isTM[i])tm_vec.push_back(0);
//      else tm_vec.push_back(1);
//    }
   
//   return kFALSE;
// }

int flavourComp3L(VecI_t& fllep) {
  const auto size = fllep.size();
  if(size>=3){
    //std::cout<<"ERROR \t Vector must be at least 3 long!"<<std::endl;
    if(fllep[0] == 1 && fllep[1] == 1 && fllep[2] == 1)return 0;
    if(fllep[0] == 1 && fllep[1] == 1 && fllep[2] == 2)return 1;
    if(fllep[0] == 1 && fllep[1] == 2 && fllep[2] == 2)return 2;
    if(fllep[0] == 2 && fllep[1] == 2 && fllep[2] == 2)return 3;
    if(fllep[0] == 2 && fllep[1] == 2 && fllep[2] == 1)return 4;
    if(fllep[0] == 2 && fllep[1] == 1 && fllep[2] == 1)return 5;
  }else if(size==2){
    if(fllep[0] == 1 && fllep[1] == 1)return 6;                                                                                                                                                                                                                    
    if(fllep[0] == 2 && fllep[1] == 2)return 7;                                                                                                                                                                                                                    
    if((fllep[0] == 1 && fllep[1] == 2) || (fllep[0] == 2 && fllep[1] == 1))return 8;                                                                                                                                                                                               
  }
  return -1;
}

bool deltaRlepjet(float lpt, float leta, float lphi, float le, VecF_t& jpt, VecF_t& jeta, VecF_t& jphi, VecF_t& je){
  const auto njet = int(jpt.size());
  TLorentzVector p2;
  TLorentzVector p1;
  double deltaR;
  double mindr = 9999;
  //for (size_t i=0; i < nlep; ++i) {
  p1.SetPtEtaPhiM(lpt, leta, lphi, le);
  for (int j=0; j < njet; ++j) {
    p2.SetPtEtaPhiM(jpt[j], jeta[j], jphi[j], je[j]);
    deltaR = p1.DeltaR(p2);
    if(deltaR < mindr){
      mindr = deltaR;
    }
    //}
  }
  //  std::cout<<"mindr = "<<mindr<<std::endl;
  return mindr;  
}

bool isSF(VecI_t& fllep) {
    if(fllep[0] == fllep[1])return kTRUE;
    return kFALSE;
}

bool isEE(VecI_t& fllep) {
    if(fllep[0] == fllep[1] && fllep[0] == 1)return kTRUE;
    return kFALSE;
}

bool isMM(VecI_t& fllep) {
    if(fllep[0] == fllep[1] && fllep[0] == 2)return kTRUE;
    return kFALSE;
}

float ComputeInvariantMass(VecF_t& pt, VecF_t& eta, VecF_t& phi, VecF_t& e) {
  TLorentzVector p1;
  TLorentzVector p2;
  p1.SetPtEtaPhiM(pt[0], eta[0], phi[0], e[0]);
  p2.SetPtEtaPhiM(pt[1], eta[1], phi[1], e[1]);
  return (p1 + p2).M();
}




float ptllboost(VecF_t& pt, VecF_t& eta, VecF_t& phi, VecF_t& e, Float_t met_et, Float_t met_phi) {

    TLorentzVector p1;
    TLorentzVector p2;
    TLorentzVector met;
    p1.SetPtEtaPhiM(pt[0], eta[0], phi[0], e[0]);
    p2.SetPtEtaPhiM(pt[1], eta[1], phi[1], e[1]);
    met.SetPtEtaPhiM(met_et, 0.0, met_phi, 0.0);
    return (met+p1+p2).Pt();
}

float costhetastar(VecF_t& pt, VecF_t& eta, VecF_t& phi, VecF_t& e) {

    TLorentzVector p1;
    TLorentzVector p2;
    p1.SetPtEtaPhiM(pt[0], eta[0], phi[0], e[0]);
    p2.SetPtEtaPhiM(pt[1], eta[1], phi[1], e[1]);
    return TMath::ATan(fabs(p1.Eta()-p2.Eta())/2.);
}

float deltaPhi_ll(VecF_t& pt, VecF_t& eta, VecF_t& phi, VecF_t& e) {

    TLorentzVector p1;
    TLorentzVector p2;
    p1.SetPtEtaPhiM(pt[0], eta[0], phi[0], e[0]);
    p2.SetPtEtaPhiM(pt[1], eta[1], phi[1], e[1]);
    return p1.DeltaPhi(p2);
}

float deltaPhi_metl(VecF_t& pt, VecF_t& eta, VecF_t& phi, VecF_t& e, Float_t met_et, Float_t met_phi) {

    TLorentzVector p1;
    TLorentzVector p2;
    TLorentzVector met;
    p1.SetPtEtaPhiM(pt[0], eta[0], phi[0], e[0]);
    p2.SetPtEtaPhiM(pt[1], eta[1], phi[1], e[1]);
    met.SetPtEtaPhiM(met_et, 0.0, met_phi, 0.0);
    
    if(p1.Pt() > p2.Pt()){
        return p1.DeltaPhi(met);
    }else{
        return p2.DeltaPhi(met);
    }
}

float deltaPhi_metll(VecF_t& pt, VecF_t& eta, VecF_t& phi, VecF_t& e, Float_t met_et, Float_t met_phi) {

  if(pt.size() < 2){
    return -999;
  }
  
  TLorentzVector p1;
  TLorentzVector p2;
  TLorentzVector dil;
  TLorentzVector met;
  p1.SetPtEtaPhiM(pt[0], eta[0], phi[0], e[0]);
  p2.SetPtEtaPhiM(pt[1], eta[1], phi[1], e[1]);

  dil = (TLorentzVector)(p1+p2);
  met.SetPtEtaPhiM(met_et, 0.0, met_phi, 0.0);
    
  return dil.DeltaPhi(met);

}

bool checkPt(VecF_t& pt, float cut1, float cut2){
    if((pt[0] > cut1 && pt[1] > cut2) || (pt[1] > cut1 && pt[0] > cut2))return kTRUE;
    return kFALSE;
}


float getET_part(VecF_t& Pt, VecF_t& M, int i){

  
    /* Calculates E_T for a given event */
    
    return sqrt(Pt[i]*Pt[i] + M[i]*M[i]);
}



float getET(float pt, float m){
    /* Calculates E_T for a given event */
    
    return sqrt(pt*pt + m*m);
}


float delta_e_T(VecF_t& Pt, VecF_t& M, int i){
    /* Calculates the transverse energy inbalance for two particles, either two leptons or two jets */

    
    
    const auto size = int(Pt.size());
      if(i > size){
        printf("delta_e_T::ERROR \t Indices %i is higher than size of vector %i\n",i,size);
        return 0;
      }
    
    float delta_e_T_j = ( getET(Pt[i-1], M[i-1]) - getET(Pt[i], M[i]) ) / 
                        ( getET(Pt[i-1], M[i-1]) + getET(Pt[i], M[i]) ) ;
    
    
    return delta_e_T_j;
}

float getRapidity(float pt, float eta, float phi, float e){
    /* Calculates the rapidity based on the pseudorapidity via Lorentz vector */

    TLorentzVector p1;
    
    p1.SetPtEtaPhiM(pt, eta, phi, e);
    
    float y = p1.Rapidity();
    return y;
    
}

float geth_L(VecF_t& pt, VecF_t& eta, VecF_t& phi, VecF_t& e, int i) {
    /* h_L is proportional to Lorentz factor, and can reflect on longitudal directions */

    
    
    const auto size = int(pt.size());
      if(i > size){
        printf("geth_L::ERROR \t Indices %i is higher than size of vector %i\n",i,size);
        return 0;
      }
    
    float y = getRapidity(pt[i], eta[i], phi[i], e[i]);
    
    return C*(cosh(y) - 1);
}

float geth(VecF_t& pt_i, VecF_t& eta_i, VecF_t& phi_i, VecF_t& e_i, VecF_t& pt_j, VecF_t& eta_j, VecF_t& phi_j, VecF_t& e_j, int i, int j){
    
    /* Similar to h_L but looks at rapidity differences between two particles, be it jets or leptons */

    
    
    const auto size_i = int(pt_i.size());
    const auto size_j = int(pt_j.size());
      if(i > size_i){
        printf("geth::ERROR \t Indices %i is higher than size of vector %i\n",j,size_j);
        return 0;
      }
      if(j > size_j){
        printf("geth::ERROR \t Indices %i is higher than size of vector %i\n",j,size_j);
        return 0;
      }
    
    
    float y_i = getRapidity(pt_i[i], eta_i[i], phi_i[i], e_i[i]);
    float y_j = getRapidity(pt_j[j], eta_j[j], phi_j[j], e_j[j]);
    
    float delta_y = y_i - y_j;
    return C*(cosh(delta_y/2) - 1);
}

float getM_T(VecF_t& pt, VecF_t& eta, VecF_t& phi, VecF_t& e, int i){
    /* Calculates the rapidity based on the pseudorapidity via Lorentz vector */
    
    /* Remember to scale with 1/sqrt(s) to get value in range [0,1] */

    
    
    const auto size = int(pt.size());
      if(i > size){
        printf("getM_T::ERROR \t Indices %i is higher than size of vector %i\n",i,size);
        return 0;
      }
    TLorentzVector p1;
    
    p1.SetPtEtaPhiM(pt[i], eta[i], phi[i], e[i]);
    
    
    return p1.Mt();
    
}


float getM(VecF_t& pt_i, VecF_t& eta_i, VecF_t& phi_i, VecF_t& e_i, VecF_t& pt_j, VecF_t& eta_j, VecF_t& phi_j, VecF_t& e_j, int i, int j){
    
    /* Similar to h_L but looks at rapidity differences between two particles, be it jets or leptons */

    

    const auto size_i = int(pt_i.size());
    const auto size_j = int(pt_j.size());
      if(i > size_i){
        printf("getM::ERROR \t Indices %i is higher than size of vector %i\n",j,size_j);
        return 0;
      }
      if(j > size_j){
        printf("getM::ERROR \t Indices %i is higher than size of vector %i\n",j,size_j);
        return 0;
      }
    
    TLorentzVector p1;
    TLorentzVector p2;
    
    
    p1.SetPtEtaPhiM(pt_i[i], eta_i[i], phi_i[i], e_i[i]);
    p2.SetPtEtaPhiM(pt_j[j], eta_j[j], phi_j[j], e_j[j]);
    
    
    return (p1 + p2).M();
}
