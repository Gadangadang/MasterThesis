import sys
import ROOT as R
from samples import configure_samples

d_samp, d_type, d_reg = configure_samples()  # False,False,True,False,False)


astyle = "/home/sgfrette/atlasrootstyle"
R.gROOT.SetMacroPath(astyle)
R.gROOT.LoadMacro("AtlasUtils.C")
R.gROOT.LoadMacro("AtlasStyle.C")
R.gROOT.LoadMacro("AtlasLabels.C")
R.SetAtlasStyle()


def myLine(x1, y1, x2, y2, color=R.kBlack, style=1, width=2):

    l = R.TLine()
    l.SetNDC()
    l.SetLineColor(color)
    l.SetLineStyle(style)
    l.DrawLine(x1, y1, x2, y2)


def myText(x, y, text, tsize=0.05, color=R.kBlack, angle=0):

    l = R.TLatex()
    l.SetTextSize(tsize)
    l.SetNDC()
    l.SetTextColor(color)
    l.SetTextAngle(angle)
    l.DrawLatex(x, y, "#bf{" + text + "}")
    l.SetTextFont(4)


def findKey(histname):
    sp = histname.split("_")
    for i in range(len(sp)):
        s = "_".join(sp[i:])
        if s in d_samp.keys():
            return s
    return ""


def getRebin(hname):
    if "MET_" in hname:
        return 4
    if "mymll_" in hname:
        return 5
    if "ptllboost" in hname:
        return 4
    if "lepPt" in hname:
        return 2
    if "jetPt" in hname:
        return 2
    return 0


class Plot:
    def __init__(p, hdic, hname="lepPt_ele_hT", bkgs=[], is1D=True, doscale=False):

        print("Plotting %s" % hname)

        p.doscale = doscale
        p.is1D = is1D
        p.is2D = not is1D
        p.rebin = 0
        p.xtit = hname
        p.intlumi = 0.0
        p.sqrts = ""
        p.datasumh = -1
        
        print(p.xtit)

        if p.is1D:
            p.plot1D(hdic, hname, bkgs)
        else:
            p.plot2D(hdic, hname, bkgs)
            
        

    def plot2D(p, hdic, hname="lepPt_ele_hT", bkgs=[]):

        p.isEff = False

        p.nTotBkg = 0.0
        p.stackorder = []
        p.dyield = {}

        # p.doScale(hdic,hname,["WmuHNL50_30G"],1.0/80000.)
        # Legend
        p.leg = R.TLegend(0.60, 0.76, 0.91, 0.90)
        p.leg.SetBorderSize(0)
        p.leg.SetTextSize(0.04)
        p.leg.SetNColumns(2)

        p.getYields(hdic, hname, bkgs)
        p.can = R.TCanvas("c1", "c1", 1000, 1000)
        # p.customise_gPad()

        if p.doscale:
            p.scaleToUnity(hdic, hname, bkgs)

        p.hstack = R.THStack()
        p.fillStack(hdic, hname, bkgs)

        histarray = p.hstack.GetStack()

        print("All histograms: %i" % histarray.GetEntries())

        for bkg in p.dyield.keys():
            if d_samp[bkg]["type"] == "bkg" and (p.dyield[bkg] / p.nTotBkg) < 0.05:
                for ha in histarray:
                    if bkg in ha.GetName():
                        histarray.Remove(ha)
                        break

        print("Histograms to plot (>0.05 of total): %i" % histarray.GetEntries())

        p.can.Divide(3, 3)
        p.can.Update()
        i = 1
        lines = [R.TLine(-1, -1, 1, 1) for i in range(len(histarray))]
        for ha in histarray:
            p.can.cd(i)
            # p.customise_gPad()
            p.customise_gPad(top=0.08, bot=0.08, left=0.1, right=0.15)
            ha.GetZaxis().SetRangeUser(0, 1)
            ha.Draw("colz")
            key = findKey(ha.GetName())
            myText(0.37, 0.87, "N(%s) = %.1f" % (key, p.dyield[key]), 0.05, R.kRed + 2)
            myLine(-1, -1, 1, 1, R.kRed + 2, 8, 2)
            i += 1

        # p.can.Update()

        # p.hstack.Draw("padscolz")

    def plot1D(p, hdic, hname="lepPt_ele_hT", bkgs=[]):

        p.isEff = False
        if "_EF_" in hname:
            p.isEff = True

        p.rebin = getRebin(hname)

        # Define canvas and pads
        p.can = R.TCanvas("", "", 1000, 1000)
        p.customise_gPad()
        if not p.isEff:
            p.pad1 = R.TPad("pad1", "", 0.0, 0.40, 1.0, 1.0)
            p.pad2 = R.TPad("pad2", "", 0.0, 0.00, 1.0, 0.4)

        print("bkgs = ", bkgs)

        # Margins used for the pads
        gpLeft = 0.17
        gpRight = 0.05
        # -------
        # PAD1
        # -------
        if not p.isEff:
            p.pad1.Draw()
            p.pad1.cd()
            p.customise_gPad(top=0.08, bot=0.01, left=gpLeft, right=gpRight)

        # Legend
        if not p.isEff:
            p.leg = R.TLegend(0.45, 0.71, 0.91, 0.90)
        else:
            p.leg = R.TLegend(0.45, 0.71, 0.91, 0.90)
        p.leg.SetBorderSize(0)
        p.leg.SetTextSize(0.03)
        p.leg.SetNColumns(2)

        p.nTotBkg = 0.0
        p.stackorder = []
        p.dyield = {}

        # p.doScale(hdic,hname,["WmuHNL50_30G"],1.0/80000.)

        p.getYields(hdic, hname, bkgs)

        p.hstack = R.THStack()
        p.fillStack(hdic, hname, reversed(p.stackorder))

        p.datastack = R.THStack()
        p.getData(hdic, hname, bkgs)

        p.signalstack = R.THStack()
        p.getSignal(hdic, hname, bkgs)

        # print("-->",p.hstack.GetNhists())
        if p.hstack.GetNhists() > 0:
            if not p.isEff:
                p.hstack.Draw("hist")
            else:
                p.hstack.Draw("nostack")
            if p.datastack.GetNhists() > 0:
                p.datasumh.Draw("same ep")
            if p.signalstack.GetNhists() > 0:
                p.signalstack.Draw("nostack same hist")
        elif p.datastack.GetNhists() > 0:
            p.datasumh.Draw("ep")
            if p.signalstack.GetNhists() > 0:
                p.signalstack.Draw("nostack same hist")
        elif p.signalstack.GetNhists() > 0:
            p.signalstack.Draw("nostack hist")
        else:
            print("Sorry there's nothing there to plot")

        p.leg.Draw()

        # p.datastack.GetXaxis().SetTitle(hname)

        # Text for ATLAS, energy, lumi, region, ntuple status
        ATL_status = "Internal"
        # lumi = 1258.27/1000.
        text_size = 0.045

        print(p.nTotBkg)
        # myText(0.22, 0.87, '#bf{#it{ATLAS}} ' + ATL_status, text_size*1.2, R.kBlack)
        myText(
            0.22,
            0.81,
            "%s TeV, %.1f  fb^{#minus1}" % (p.sqrts, float(p.intlumi) / 1000.0),
            text_size * 1.1,
            R.kBlack,
        )
        # myText(0.22, 0.77, sig_reg, text_size*0.7, R.kBlack)
        # myText()
        # myText(0.22, 0.73, NTUP_status, text_size*0.7, kGray+1)

        if not p.isEff:
            myText(0.22, 0.69, "N_bkg: {:.0f}".format(p.nTotBkg), 0.040, R.kBlack)

        # print(p.hstack.GetXaxis().GetTitle())
        xtitle = p.xtit
        ytitle = "Events" if not p.isEff else "Efficieny"
        IsLogY = True
        enlargeYaxis = False
        scaling = "False"

        if not p.isEff:
            p.pad1.SetLogy(IsLogY)

        try:
            maxbin = (
                p.hstack.GetStack()
                .Last()
                .GetBinContent(p.hstack.GetStack().Last().GetMaximumBin())
            )
            print(f"xtitle = {xtitle}")
            p.customise_axes(
                p.hstack,
                xtitle,
                ytitle,
                1.1,
                IsLogY,
                enlargeYaxis,
                maxbin,
                scaling == "True",
            )
        except:
            maxbin = 0
            print(f"xtitle = {xtitle}")
            p.customise_axes(
                p.datastack,
                xtitle,
                ytitle,
                1.1,
                IsLogY,
                enlargeYaxis,
                maxbin,
                scaling == "True",
            )

        # if not p.isEff:
        #  myText(0.77, 0.47, 'N(Bkg) = %.1f'%(p.nTotBkg), 0.025, R.kBlack)

        p.ratio = R.TH1D()
        # if (p.hstack.GetStack().Last() != None or p.datastack.GetStack().Last() != None):
        if not p.isEff:
            try:
                p.getRatio(p.hstack.GetStack().Last(), p.datastack.GetStack().Last())
            except:
                print("Could not get ratio!")

        # -------
        # PAD2
        # -------
        p.can.cd()
        if not p.isEff:
            p.pad2.Draw()
            p.pad2.cd()
            p.customise_gPad(top=0.05, bot=0.39, left=gpLeft, right=gpRight)
            # customise_gPad(top=0, bot=0.39, left=gpLeft, right=gpRight) # joins upper and lower plot
            p.pad2.SetGridy()
            p.ratio.Draw("e0p")

            xtitle = p.xtit
            ytitle = "Ratio Data/MC"
            IsLogY = True
            enlargeYaxis = False
            scaling = "False"

            maxbin = p.ratio.GetBinContent(p.ratio.GetMaximumBin())
            print(f"xtitle = {xtitle}")
            p.customise_axes(
                p.ratio,
                xtitle,
                ytitle,
                1.1,
                IsLogY,
                enlargeYaxis,
                maxbin,
                scaling == "True",
            )

            p.can.Update()

            # p.can.SaveAs("plots/%s.pdf"%hname)

    def scaleToUnity(p, histo, hkey, procs):
        for k in procs:
            if not hkey + "_%s" % k in histo.keys():
                continue
            if p.is1D:
                histo[hkey + "_%s" % k].Scale(
                    1.0
                    / (
                        histo[hkey + "_%s" % k].Integral(
                            0, histo[hkey + "_%s" % k].GetNbinsX() + 1
                        )
                    )
                )
            else:
                histo[hkey + "_%s" % k].Scale(
                    1.0
                    / histo[hkey + "_%s" % k].Integral(
                        0,
                        histo[hkey + "_%s" % k].GetNbinsX() + 1,
                        0,
                        histo[hkey + "_%s" % k].GetNbinsY() + 1,
                    )
                )

    def getYields(p, histo, hkey, procs):
        for k in procs:
            newkey = hkey + "_%s" % k
            # print("newkey",newkey)
            if p.isEff:
                newkey = hkey.replace("_EF_", "_SG_") + "_%s" % k
            # print(histo.keys())
            if not newkey in histo.keys():
                print("continuing getYields")
                continue
            if p.is1D:
                p.dyield[k] = histo[newkey].Integral(0, histo[newkey].GetNbinsX() + 1)
            else:
                p.dyield[k] = histo[newkey].Integral(
                    0, histo[newkey].GetNbinsX() + 1, 0, histo[newkey].GetNbinsY() + 1
                )
            if d_samp[k]["type"] == "bkg":
                p.nTotBkg += p.dyield[k]
        newdict = p.dyield.copy()
        while True:
            maxi = -99999
            maxkey = ""
            for k in newdict.keys():
                if not d_samp[k]["type"] == "bkg":
                    continue
                if newdict[k] > maxi:
                    maxkey = k
                    maxi = newdict[k]
            if not maxkey:
                break
            if maxkey in p.stackorder:
                break
            # print("maxkey = ",maxkey)
            p.stackorder.append(maxkey)
            ret = newdict.pop(maxkey, None)
            if ret == None:
                break

    def doScale(p, histo, hkey, procs, fac=1.0):
        for k in procs:
            if d_samp[k]["type"] == "data":
                continue
            histo[hkey + "_%s" % k].Scale(fac)

    def getRatio(p, h1, h2):
        p.ratio = h2.Clone("hRatio")
        p.ratio.Sumw2()
        p.ratio.Divide(h1)
        # p.ratio.SetLineColor(R.kGray+2)
        for ibin in range(1, p.ratio.GetNbinsX() + 1):
            if p.ratio.GetBinContent(ibin) == 0:
                p.ratio.SetBinContent(ibin, -10)
        p.ratio.SetLineWidth(2)
        p.ratio.SetMarkerStyle(21)
        
   

    def fillStack(p, histo, hkey, procs):
        print("fillstack")
     
        zees = ['Zeejets1', 'Zeejets2', 'Zeejets3', 'Zeejets4', 'Zeejets5', 'Zeejets6', 'Zeejets7', 'Zeejets8', 'Zeejets9', 'Zeejets10', 'Zeejets11', 'Zeejets12', 'Zeejets13', 'Zeejets14', 'Zeejets15']
        zmms = ['Zmmjets1', 'Zmmjets2', 'Zmmjets3', 'Zmmjets4', 'Zmmjets5', 'Zmmjets6', 'Zmmjets7', 'Zmmjets8', 'Zmmjets9', 'Zmmjets10', 'Zmmjets11', 'Zmmjets12', 'Zmmjets13']
        
        print("Stack")
        print(p.stackorder)
        
        zee_channel_count = 0
        zmm_channel_count = 0
        
        tot_zee = 0
        tot_zmm = 0
        
        zee_check = 0
        zmm_check = 0
        for k in procs:
            
            if p.is1D and not d_samp[k]["type"] == "bkg":
                print("continue")
                continue
            if not hkey + "_%s" % k in histo.keys():
                print("Could not find key %s in histo dictionary" % (hkey + "_%s" % k))
                continue

            pc_yield = 0.0
            
            
            if float(p.nTotBkg) != 0:
                if k in zees:
                    tot_zee += p.dyield[k]
                    zee_channel_count += 1
                    choice = tot_zee
                elif k in zmms:
                    tot_zmm += p.dyield[k]
                    zmm_channel_count += 1
                    choice = tot_zmm
                else:
                    choice = p.dyield[k]
                    
                pc_yield = 100 * (choice / float(p.nTotBkg))
            
            try:
                # print("Adding %s_%s"%(hkey,k))
                
                if p.rebin:
                    histo[hkey + "_%s" % k] = histo[hkey + "_%s" % k].Rebin(p.rebin)
                
                if k in zees:
                    if zee_check == 0:
                        p.zeejetsh = histo[hkey + "_%s" % k].Clone(hkey + "_%s_SUM" % k)
                        zee_check += 1
                        
                    
                    
                    p.zeejetsh.Add(histo[hkey + "_%s" % k].GetValue())
                    
                    if zee_channel_count == 15:
                        
                        leg_txt = "{0} ({1:.1f}%)".format(d_samp[k]["leg"], pc_yield)
                        print(leg_txt)
                        p.leg.AddEntry(histo[hkey + "_%s" % k], leg_txt, "lpf")
                        
                        p.hstack.Add(p.zeejetsh)
                
                elif k in zmms:
                    if zmm_check == 0:
                        
                        p.zmmjetsh = histo[hkey + "_%s" % k].Clone(hkey + "_%s_SUM" % k)
                        zmm_check += 1
                        print("created zmm histo")
                    
                    
                    p.zmmjetsh.Add(histo[hkey + "_%s" % k].GetValue())
                    
                    if zmm_channel_count == 13:
                        
                        leg_txt = "{0} ({1:.1f}%)".format(d_samp[k]["leg"], pc_yield)
                        print(leg_txt)
                        p.leg.AddEntry(histo[hkey + "_%s" % k], leg_txt, "lpf")
                        p.hstack.Add(p.zmmjetsh)
                    
                else:
                    p.hstack.Add(histo[hkey + "_%s" % k])
                    leg_txt = "{0} ({1:.1f}%)".format(d_samp[k]["leg"], pc_yield)
                    print(leg_txt)
                    p.leg.AddEntry(histo[hkey + "_%s" % k], leg_txt, "lpf")
                
                
                # print("xtitle = %s"%p.xtit)
            except:
               
                # print("Adding %s_%s"%(hkey,k))
                
                if k in zees:
                    if zee_check == 0:
                        p.zeejetsh = histo[hkey + "_%s" % k].Clone(hkey + "_%s_SUM" % k)
                        zee_check += 1
                        print("created zee histo")
                    
                    
                    p.zeejetsh.Add(histo[hkey + "_%s" % k].GetValue())
                    
                    
                    if zee_channel_count == 15:
                        
                        leg_txt = "{0} ({1:.1f}%)".format(d_samp[k]["leg"], pc_yield)
                        print(leg_txt)
                        p.leg.AddEntry(histo[hkey + "_%s" % k].GetValue(), leg_txt, "lpf")
                        p.hstack.Add(p.zeejetsh)
                
                elif k in zmms:
                    if zmm_check == 0:
                        p.zmmjetsh = histo[hkey + "_%s" % k].Clone(hkey + "_%s_SUM" % k)
                        zmm_check += 1
                        print("created zmm histo")
                    
                    
                    p.zmmjetsh.Add(histo[hkey + "_%s" % k].GetValue())
                    
                    if zmm_channel_count == 13:
                        
                        leg_txt = "{0} ({1:.1f}%)".format(d_samp[k]["leg"], pc_yield)
                        print(leg_txt)
                        p.leg.AddEntry(histo[hkey + "_%s" % k].GetValue(), leg_txt, "lpf")
                        p.hstack.Add(p.zmmjetsh)
                    
                else:
                    p.hstack.Add(histo[hkey + "_%s" % k].GetValue())
                    leg_txt = "{0} ({1:.1f}%)".format(d_samp[k]["leg"], pc_yield)
                    print(leg_txt)
                    p.leg.AddEntry(histo[hkey + "_%s" % k].GetValue(), leg_txt, "lpf")
              

    def getData(p, histo, hkey, procs):
        tot_data = 0
        for k in procs:

            if not d_samp[k]["type"] == "data":
                continue
            if not hkey + "_%s" % k in histo.keys():
                continue
            
            # try:
            if p.rebin:
                histo[hkey + "_%s" % k] = histo[hkey + "_%s" % k].Rebin(p.rebin)

            p.datastack.Add(histo[hkey + "_%s" % k].GetValue())
            if not type(p.datasumh) is R.TH1D:
                p.datasumh = histo[hkey + "_%s" % k].Clone(hkey + "_%s_SUM" % k)
            else:
                tot_data += p.dyield[k]

                p.datasumh.Add(histo[hkey + "_%s" % k].GetValue())
                if k in ["data18"]:
                    leg_txt = "{0} ({1:.0f} Events)".format("Data", tot_data)
                    p.leg.AddEntry(histo[hkey + "_%s" % k].GetValue(), leg_txt, "lp")
                # except:
                #    p.datastack.Add(histo[hkey+"_%s"%k].GetValue())
                # p.datasumh.Add(histo[hkey+"_%s"%k].GetValue()))
                # p.leg.AddEntry(histo[hkey+"_%s"%k].GetValue(),leg_txt,"lp")
            p.intlumi += d_samp[k]["lumi"]
            p.sqrts = str(d_samp[k]["sqrts"])

    def getSignal(p, histo, hkey, procs):
        for k in procs:
            if not d_samp[k]["type"] == "sig":
                continue
            if not hkey + "_%s" % k in histo.keys():
                continue
            if not p.isEff:
                leg_txt = "{0} ({1:.0f} Events)".format(d_samp[k]["leg"], p.dyield[k])
            else:
                leg_txt = "{0}".format(d_samp[k]["leg"])
            try:
                p.signalstack.Add(histo[hkey + "_%s" % k])
                p.leg.AddEntry(histo[hkey + "_%s" % k], leg_txt, "lp")
            except:
                p.signalstack.Add(histo[hkey + "_%s" % k].GetValue())
                p.leg.AddEntry(histo[hkey + "_%s" % k].GetValue(), leg_txt, "lp")

    # Function for customising the gPad (gPad points to the current pad, and one can use gPad to set attributes of the current pad)

    def customise_gPad(p, top=0.03, bot=0.15, left=0.17, right=0.08):

        R.gPad.Update()

        R.gStyle.SetTitleFontSize(0.0)

        # gPad margins
        R.gPad.SetTopMargin(top)
        R.gPad.SetBottomMargin(bot)
        R.gPad.SetLeftMargin(left)
        R.gPad.SetRightMargin(right)

        R.gStyle.SetOptStat(0)  # Hide usual stats box

        R.gPad.Update()

    # Funcion for customising axes
    def customise_axes(
        p,
        hist,
        xtitle,
        ytitle,
        scaleFactor=1.1,
        IsLogY=False,
        enlargeYaxis=False,
        maxbin=10,
        scaling=False,
    ):

        # print(type(hist))
        # Set a universal text size
        text_size = 45

        R.TGaxis.SetMaxDigits(4)

        ##################################
        # X axis
        xax = hist.GetXaxis()

        # Precision 3 Helvetica (specify label size in pixels)
        xax.SetLabelFont(43)
        xax.SetTitleFont(43)
        # xax.SetTitleFont(13) # times

        xax.SetTitle(xtitle)
        xax.SetTitleSize(text_size)

        print("ytitle  = ", ytitle)

        # Top panel
        if "Events" in ytitle:
            xax.SetLabelSize(0)
            xax.SetLabelOffset(0.02)
            xax.SetTitleOffset(1.4)
            xax.SetTickSize(0.04)
        # Bottom panel
        else:
            xax.SetLabelSize(text_size - 7)
            xax.SetLabelOffset(0.03)
            xax.SetTitleOffset(3.0)
            xax.SetTickSize(0.08)

        # xax.SetRangeUser(0,2000)
        # xax.SetNdivisions(-505)

        R.gPad.SetTickx()

        ##################################
        # Y axis
        yax = hist.GetYaxis()

        # Precision 3 Helvetica (specify label size in pixels)
        yax.SetLabelFont(43)
        yax.SetTitleFont(43)

        yax.SetTitle(ytitle)
        yax.SetTitleSize(text_size)
        yax.SetTitleOffset(1.8)

        yax.SetLabelOffset(0.015)
        yax.SetLabelSize(text_size - 7)

        ymax = hist.GetMaximum()
        ymin = hist.GetMinimum()

        # print(yax.GetTitle()
        # print('ymax       = ', ymax)
        # print('SF         = ', scaleFactor )
        # print('ymax x SF  = ', ymax*scaleFactor)
        # print('ymin       = ', ymin)
        # print('ymin x 0.9 = ', ymin*scaleFactor)
        # if ymin == 0.0:
        #     print 'ymin = 0.0'

        # Top events panel
        if "Events" in ytitle:
            yax.SetNdivisions(505)
            yax.SetTitleOffset(1.4)
            if IsLogY:
                if enlargeYaxis:
                    ymax *= 2 * 10**10
                    ymin = 1
                else:
                    # ymax = 3 * 10 ** 4
                    # ymin = 0.5
                    ymax *= 10 * 10
                    ymin = 1
                # if scaling:
                #    hist.SetMaximum(1.0)
                # else:
                hist.SetMaximum(ymax)
                hist.SetMinimum(ymin)
            else:
                # if scaling:
                #    hist.SetMaximum(1.0)
                # else:
                hist.SetMaximum(ymax * scaleFactor)
                hist.SetMinimum(0.0)
        elif "Efficiency" in ytitle:
            yax.SetNdivisions(505)
            hist.SetMaximum(1.2)
            hist.SetMinimum(0.0)
        # Bottom panel
        elif "Ratio" in ytitle:
            yax.SetNdivisions(505)
            yax.SetTitleOffset(1.2)
            # Dynamic
            if ymax * scaleFactor > 10:
                hist.SetMaximum(5)
            else:
                hist.SetMaximum(ymax * scaleFactor)
            if ymin * 0.9 < -1:
                hist.SetMinimum(-2)  # ymin*0.9)
            else:
                hist.SetMinimum(ymin * 0.9)
            # Fixed
            # hist.SetMinimum(-0.5)
            # hist.SetMaximum(2.5)

        R.gPad.SetTicky()

        R.gPad.Update()
