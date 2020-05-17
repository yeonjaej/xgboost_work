import uproot
import pandas
import numpy as np
import xgboost


file_train_sig = uproot.open("/pnfs/uboone/persistent/users/yjwa/ub_nnbar_bdt_tree/nnbar/31220041_0/usinghandlesana_out.root")
file_train_bkg = uproot.open("/pnfs/uboone/persistent/users/yjwa/ub_nnbar_bdt_tree/corsika/31220052_0/usinghandlesana_out.root")

dir_train_sig = file_train_sig["usinghandlesanamod"]
dir_train_bkg = file_train_bkg["usinghandlesanamod"]
tree_train_sig = dir_train_sig["out_tree"]
tree_train_bkg = dir_train_bkg["out_tree"]

#tree_train = file_train["out_tree"]
#tree_test = file_test["tree_name"]
#tree_train_and_test = file_train_and_test["tree_name"]

# function here just so I don't repeat this four times in the code

def getit(tf):
    # select which branches not to import
    df = tf.pandas.df(lambda branch:branch.name == b'slice_score' or branch.name == b'pf_energy_track' or branch.name == b'pf_energy_shower' or branch.name == b'leading_track_E' or branch.name == b'leading_track_mu_chi' or branch.name == b'leading_track_pi_chi' or branch.name == b'leading_shower_E' or branch.name == b'num_slice_tracks' or branch.name == b'num_slice_showers')
    # create new branches based on other branches
    #df['sum_nhits_1'] = df.loc[:, ['nhits1_p0','nhits1_p1','nhits1_p2']].sum(axis=1)
    #df['sum_nhits_2'] = df.loc[:, ['nhits2_p0','nhits2_p1','nhits2_p2']].sum(axis=1)
    #df['sum_nhits'] = df.loc[:, ['sum_nhits_1','sum_nhits_2']].sum(axis=1)
    return df

data_train_sig = getit(tree_train_sig)
data_train_bkg = getit(tree_train_bkg)
#data_test_sig = getit(tree_test)

train_vs_test_fraction = 0.8
#data_train_and_test = getit(tree_train_and_test)

data_train_sig = data_train_sig[int(len(data_train_sig)*train_vs_test_fraction):]
data_test_sig = data_train_sig[:int(len(data_train_sig)*train_vs_test_fraction)]

data_train_bkg = data_train_bkg[int(len(data_train_bkg)*train_vs_test_fraction):]
data_test_bkg = data_train_bkg[:int(len(data_train_bkg)*train_vs_test_fraction)]

# 1 = signal, 0 = background
labels = [1]*len(data_train_sig) + [0]*len(data_train_bkg)

xgb_train = xgboost.DMatrix(pandas.concat([data_train_sig, data_train_bkg]), label=labels)
xgb_test_sig = xgboost.DMatrix(data_test_sig, label=[1]*len(data_test_sig))
xgb_test_bkg = xgboost.DMatrix(data_test_bkg, label=[0]*len(data_test_bkg))

# watchlist so that you can monitor the performance of the training by iterations
watchlist = [(xgb_train, 'train'), (xgb_test_sig, 'test_sig'),(xgb_test_bkg,'test_bkg')]
# paramters of the training, see documentation
param = {'booster': 'dart',
        'max_depth':6,
        'eta':0.3,
        'objective':'binary:logistic',
        #'eval_metric':'auc', 
        #'subsample':0.5,
        'tree_method':'hist',
        #'scale_pos_weight': float(len(data_cos_train))/float(len(data_sig_train)),
        'rate_drop': 0.1,
         'skip_drop': 0.5 }
num_round = 300

bdt = xgboost.train(param, xgb_train, num_round, watchlist)

# save model so you can load it later
bdt.save_model('model.json')

results_sig = bdt.predict(xgb_test_sig)
results_bkg = bdt.predict(xgb_test_bkg)

importance_w= xgboost.plot_importance(bdt)
importance_w.figure.savefig('importance_weight.png')
importance_g= xgboost.plot_importance(bdt,importance_type='gain')
importance_g.figure.savefig('importance_gain.png')
importance_c= xgboost.plot_importance(bdt,importance_type='cover')
importance_c.figure.savefig('importance_cover.png')

#tree=xgboost.plot_tree(bdt)
#tree.figure.savefig('plot_tree.eps', format='eps', dpi=10000)
tree2=xgboost.plot_tree(bdt,num_trees=2)
tree2.figure.savefig('plot_tree2.eps', format='eps', dpi=1000)
# if you want to load the model later
bdt = xgboost.Booster()
bdt.load_model('model.json')

# if you need to store back into a tree
with uproot.recreate("bdt_output_sig.root") as f:
        f["tree"] = uproot.newtree({"bdt_result": "float64"})
        f["tree"].extend({"bdt_result": results_sig})
        f["tree_bkg"] = uproot.newtree({"bdt_result_bkg": "float64"})
        f["tree_bkg"].extend({"bdt_result_bkg": results_bkg})
#        f["tree"].extend({"plot_slice":plot_slice})

#plot_slice= bdt.plot_importance(slice_score)
