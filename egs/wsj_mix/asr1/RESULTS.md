# WSJ-2mix Result
## word level rnnlm with / without speaker parallel attention
## CER
| dataset | Snt | Corr | Sub | Del | Ins | Err |
| :-----: | :-: | :--: | :-: | :-: | :-: | :-: |
| exp/tr_pytorch_train_multispkr_spafalse_no_preprocess_delta/decode_cv_decode_lm_word65000_model.acc.best | 503 | 85377 | 6581 | 4890 | 3982 | 15.96 |
| exp/tr_pytorch_train_multispkr_spafalse_no_preprocess_delta/decode_tt_decode_lm_word65000_model.acc.best | 333 | 60849 | 3537 | 2695 | 1920 | 12.15 |
| exp/tr_pytorch_train_multispkr_spatrue_no_preprocess_delta/decode_cv_decode_lm_word65000_model.acc.best | 503 | 85388 | 5927 | 5533 | 2875 | 14.80 |
| exp/tr_pytorch_train_multispkr_spatrue_no_preprocess_delta/decode_tt_decode_lm_word65000_model.acc.best | 333 | 61630 | 3176 | 2275 | 1842 | 10.87 |
| exp/tr_pytorch_train_multispkr_transformer_no_preprocess_delta/decode_cv_decode_lm_word65000_model.last10.avg.best | 503 | 89863 | 3417 | 3568 | 1704 | 8.97 |
| exp/tr_pytorch_train_multispkr_transformer_no_preprocess_delta/decode_tt_decode_lm_word65000_model.last10.avg.best | 333 | 63659 | 1720 | 1702 | 942 | 6.51 |
## WER
| dataset | Snt | Corr | Sub | Del | Ins | Err |
| :-----: | :-: | :--: | :-: | :-: | :-: | :-: |
| exp/tr_pytorch_train_multispkr_spafalse_no_preprocess_delta/decode_cv_decode_lm_word65000_model.acc.best | 503 | 12691 | 3169 | 566 | 651 | 26.70 |
| exp/tr_pytorch_train_multispkr_spafalse_no_preprocess_delta/decode_tt_decode_lm_word65000_model.acc.best | 333 | 9350 | 1677 | 291 | 308 | 20.11 |
| exp/tr_pytorch_train_multispkr_spatrue_no_preprocess_delta/decode_cv_decode_lm_word65000_model.acc.best | 503 | 12817 | 2921 | 688 | 475 | 24.86 |
| exp/tr_pytorch_train_multispkr_spatrue_no_preprocess_delta/decode_tt_decode_lm_word65000_model.acc.best | 333 | 9557 | 1501 | 260 | 308 | 18.28 |
| exp/tr_pytorch_train_multispkr_transformer_no_preprocess_delta/decode_cv_decode_lm_word65000_model.last10.avg.best | 503 | 13996 | 1989 | 441 | 317 | 16.72 |
| exp/tr_pytorch_train_multispkr_transformer_no_preprocess_delta/decode_tt_decode_lm_word65000_model.last10.avg.best | 333 | 10094 | 1055 | 169 | 200 | 12.58 |

The mixture scheme is in the local/wsj_mix_scheme.tar.gz
Click here to get the [pretrained model without speaker parallel attention](https://drive.google.com/open?id=11SWTPG5ggMHtqucHDTeWpNCRXrYMw4SZ).
Click here to get the [pretrained model with speaker parallel attention]().
Click here to get the [pretrained model with transformer](https://drive.google.com/open?id=1E-NzCzrgoFlPC--eOH4NYdepYljYAZMt).

# WSJ0-2mix
## CER
| dataset | Snt | Corr | Sub | Del | Ins | Err |
| :-----: | :-: | :--: | :-: | :-: | :-: | :-: |
| exp/tr_pytorch_train_multispkr_spafalse_no_preprocess_delta/decode_tt_decode_lm_word65000_model.acc.best | 3000 | 531894 | 36189 | 30213 | 18161 | 14.13 |
| exp/tr_pytorch_train_multispkr_transformer_no_preprocess_delta/decode_tt_decode_lm_word65000_model.last10.avg.best | 3000 | 553383 | 23455 | 21458 | 11304 | 9.40 |
## WER
| dataset | Snt | Corr | Sub | Del | Ins | Err |
| :-----: | :-: | :--: | :-: | :-: | :-: | :-: |
| exp/tr_pytorch_train_multispkr_spafalse_no_preprocess_delta/decode_tt_decode_lm_word65000_model.acc.best | 3000 | 79432 | 15538 | 3643 | 2744 | 22.23 |
| exp/tr_pytorch_train_multispkr_transformer_no_preprocess_delta/decode_tt_decode_lm_word65000_model.last10.avg.best | 3000 | 83533 | 12552 | 2528 | 2394 | 17.72 |

Click here to get the [pretrained model without speaker parallel attention](https://drive.google.com/open?id=1yiinAMHczS3JpK5b5bnt-BKqH1AMTFjH).
Click here to get the [pretrained model with speaker parallel attention](https://drive.google.com/open?id=1xm2W1AXBgnccq-AFkp5-sDe2R0X8CMNx).
