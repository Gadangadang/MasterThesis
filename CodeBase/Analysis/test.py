 """
        Zeejets = self.recon_err_back[
            np.where(self.data_structure.val_categories == "Zeejets")[0]
        ]
        Zmmjets = self.recon_err_back[
            np.where(self.data_structure.val_categories == "Zmmjets")[0]
        ]
        Zttjets = self.recon_err_back[
            np.where(self.data_structure.val_categories == "Zttjets")[0]
        ]
        diboson2L = self.recon_err_back[
            np.where(self.data_structure.val_categories == "diboson2L")[0]
        ]
        diboson3L = self.recon_err_back[
            np.where(self.data_structure.val_categories == "diboson3L")[0]
        ]
        diboson4L = self.recon_err_back[
            np.where(self.data_structure.val_categories == "diboson4L")[0]
        ]
        triboson = self.recon_err_back[
            np.where(self.data_structure.val_categories == "triboson")[0]
        ]
        higgs = self.recon_err_back[
            np.where(self.data_structure.val_categories == "higgs")[0]
        ]
        singletop = self.recon_err_back[np.where(self.data_structure.val_categories == "singletop")[0]]
        
        topOther = self.recon_err_back[
            np.where(self.data_structure.val_categories == "topOther")[0]
        ]
        Wjets = self.recon_err_back[
            np.where(self.data_structure.val_categories == "Wjets")[0]
        ]
        ttbar = self.recon_err_back[
            np.where(self.data_structure.val_categories == "ttbar")[0]
        ]

        Zeejets_w = self.data_structure.weights_val[
            np.where(self.data_structure.val_categories == "Zeejets")[0]
        ]
        Zmmjets_w = self.data_structure.weights_val[
            np.where(self.data_structure.val_categories == "Zmmjets")[0]
        ]
        Zttjets_w = self.data_structure.weights_val[
            np.where(self.data_structure.val_categories == "Zttjets")[0]
        ]
        diboson2L_w = self.data_structure.weights_val[
            np.where(self.data_structure.val_categories == "diboson2L")[0]
        ]
        diboson3L_w = self.data_structure.weights_val[
            np.where(self.data_structure.val_categories == "diboson3L")[0]
        ]
        diboson4L_w = self.data_structure.weights_val[
            np.where(self.data_structure.val_categories == "diboson4L")[0]
        ]
        triboson_w = self.data_structure.weights_val[
            np.where(self.data_structure.val_categories == "triboson")[0]
        ]
        higgs_w = self.data_structure.weights_val[
            np.where(self.data_structure.val_categories == "higgs")[0]
        ]
        singletop_w = self.data_structure.weights_val[
            np.where(self.data_structure.val_categories == "singletop")[0]
        ]
        topOther_w = self.data_structure.weights_val[
            np.where(self.data_structure.val_categories == "topOther")[0]
        ]
        Wjets_w = self.data_structure.weights_val[
            np.where(self.data_structure.val_categories == "Wjets")[0]
        ]
        ttbar_w = self.data_structure.weights_val[
            np.where(self.data_structure.val_categories == "ttbar")[0]
        ]

        histo_atlas = [
            Zeejets,
            Zmmjets,
            Zttjets,
            diboson2L,
            diboson3L,
            diboson4L,
            higgs,
            singletop,
            topOther,
            Wjets,
            triboson,
            
        ] #ttbar,
        weight_atlas_data = [
            Zeejets_w,
            Zmmjets_w,
            Zttjets_w,
            diboson2L_w,
            diboson3L_w,
            diboson4L_w,
            higgs_w,
            singletop_w,
            topOther_w,
            Wjets_w,
            triboson_w,
            
        ] #ttbar_w,
        
        """
        
      