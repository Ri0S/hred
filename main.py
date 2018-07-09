import argparse
import time
import math
import torch.nn.init as init
import torch.optim as optim
import os

from modules import *
from util import *
from collections import Counter

use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
np.random.seed(123)
if use_cuda:
    torch.cuda.manual_seed(123)


def init_param(model):
    for name, param in model.named_parameters():
        if 'embed' in name:
            continue
        elif ('rnn' in name or 'lm' in name) and len(param.size()) >= 2:
            init.orthogonal(param)
        else:
            init.normal(param, 0, 0.01)


def clip_gnorm(model):
    for name, p in model.named_parameters():
        param_norm = p.grad.data.norm()
        if param_norm > 1:
            p.grad.data.mul_(1 / param_norm)


def train(options, model):
    model.train()
    optimizer = optim.Adam(model.parameters(), options.lr)
    if os.path.isfile(options.name):
        model.load_state_dict(torch.load(options.name))
    else:
        init_param(model)

    if options.toy:
        train_dataset, valid_dataset = MovieTriples('train', 1000), MovieTriples('valid', 100)
    else:
        train_dataset, valid_dataset = MovieTriples('train', options.bt_siz), MovieTriples('valid', options.bt_siz)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                                  collate_fn=custom_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True,
                                  collate_fn=custom_collate_fn)

    # print("Training set {} Validation set {}".format(len(train_dataset), len(valid_dataset)))

    criteria = nn.CrossEntropyLoss(ignore_index=0, size_average=False)
    if use_cuda:
        criteria.cuda()

    best_vl_loss, patience, batch_id = 10000, 0, 0
    for i in range(options.epoch):
        if patience == options.patience:
            break
        tr_loss, tlm_loss, num_words = 0, 0, 0
        strt = time.time()
        for i_batch, sample_batch in enumerate(tqdm(train_dataloader)):
            new_tc_ratio = 2100.0 / (2100.0 + math.exp(batch_id / 2100.0))
            model.dec.set_tc_ratio(new_tc_ratio)
            trainData = sample_batch[:-1]
            gold = sample_batch[-1]
            preds, lmpreds = model(trainData, gold)

            preds = preds[:, :-1, :].contiguous().view(-1, preds.size(2))
            gold = gold[:, 1:].contiguous().view(-1)
            loss = criteria(preds, gold)
            target_toks = gold.ne(0).long().sum().data[0]

            num_words += target_toks
            tr_loss += loss.data[0]
            loss = loss / target_toks
            print(' epoch', i, 'batch', i_batch, 'loss:', loss.data[0])

            optimizer.zero_grad()
            loss.backward()

            clip_gnorm(model)
            optimizer.step()

            batch_id += 1
        print('total loss:', tr_loss)
        print()
        torch.save(model.state_dict(), options.name[:-3] + str(int(options.name[-3:]) + i + 1).zfill(3))

        vl_loss = calc_valid_loss(valid_dataloader, criteria, model)
        print("Training loss {} lm loss {} Valid loss {}".format(tr_loss / num_words, tlm_loss / num_words, vl_loss))
        print("epoch {} took {} mins".format(i + 1, (time.time() - strt) / 60.0))
        print("tc ratio", model.dec.get_tc_ratio())
        if vl_loss < best_vl_loss or options.toy:
            best_vl_loss = vl_loss
            patience = 0
        else:
            patience += 1


def load_model_state(mdl, fl):
    saved_state = torch.load(fl)
    mdl.load_state_dict(saved_state)


def generate(model, ses_encoding, options):
    diversity_rate = 2
    antilm_param = 10
    beam = options.beam

    n_candidates, final_candids = [], []
    candidates = [([1], 0, 0)]
    gen_len, max_gen_len = 1, 100

    while gen_len <= max_gen_len:
        for c in candidates:
            seq, pts_score, pt_score = c[0], c[1], c[2]
            _target = Variable(torch.LongTensor([seq]), volatile=True)
            dec_o, dec_lm = model.dec([ses_encoding, _target, [len(seq)]])
            dec_o = dec_o[:, :, :-1]

            op = F.log_softmax(dec_o, 2, 5)
            op = op[:, -1, :]
            topval, topind = op.topk(beam, 1)

            if options.lm:
                dec_lm = dec_lm[:, :, :-1]
                lm_op = F.log_softmax(dec_lm, 2, 5)
                lm_op = lm_op[:, -1, :]

            for i in range(beam):
                ctok, cval = topind.data[0, i], topval.data[0, i]
                if options.lm:
                    uval = lm_op.data[0, ctok]
                    if dec_lm.size(1) > antilm_param:
                        uval = 0.0
                else:
                    uval = 0.0

                if ctok == 2:
                    list_to_append = final_candids
                else:
                    list_to_append = n_candidates

                list_to_append.append((seq + [ctok], pts_score + cval - diversity_rate * (i + 1), pt_score + uval))

        n_candidates.sort(key=lambda temp: sort_key(temp, options.mmi), reverse=True)
        candidates = copy.copy(n_candidates[:beam])
        n_candidates[:] = []
        gen_len += 1

    final_candids = final_candids + candidates
    final_candids.sort(key=lambda temp: sort_key(temp, options.mmi), reverse=True)

    return final_candids[:beam]


def sort_key(temp, mmi):
    if mmi:
        lambda_param = 0.25
        return temp[1] - lambda_param * temp[2] + len(temp[0]) * 0.1
    else:
        return temp[1] / len(temp[0]) ** 0.7


def get_sent_ll(u3, u3_lens, model, criteria, ses_encoding):
    preds, _ = model.dec([ses_encoding, u3, u3_lens])
    preds = preds[:, :-1, :].contiguous().view(-1, preds.size(2))
    u3 = u3[:, 1:].contiguous().view(-1)
    loss = criteria(preds, u3).data[0]
    target_toks = u3.ne(0).long().sum().data[0]
    return -1 * loss / target_toks


def inference_beam(dataloader, model, inv_dict, options):
    criteria = nn.CrossEntropyLoss(ignore_index=0, size_average=False)
    if use_cuda:
        criteria.cuda()

    cur_tc = model.dec.get_teacher_forcing()
    model.dec.set_teacher_forcing(True)
    fout = open(options.name + "_beam" + str(options.beam) + "_result.txt", 'w', encoding='utf-8')
    model.load_state_dict(torch.load(options.name))

    model.eval()

    for i_batch, sample_batch in enumerate(dataloader):

        testData = sample_batch[:-1]
        gold = sample_batch[-1]
        us = testData.transpose(0, 1).contiguous().clone()
        qu_seq = model.base_enc(us.view(-1, us.size(2))).view(us.size(0), us.size(1), -1)
        final_session_o = model.ses_enc(qu_seq, None)

        for k in range(testData.size(1)):
            sent = generate(model, final_session_o[k, :, :].unsqueeze(0), options)
            pt = tensor_to_sent(sent, inv_dict)
            gt = tensor_to_sent(gold[k, :].unsqueeze(0).data.cpu().numpy(), inv_dict, True)
            if not options.pretty:
                print(pt)
                print("Ground truth {} {} \n".format(gt, get_sent_ll(gold[k, :].unsqueeze(0), model, criteria,
                                                                     final_session_o)))
            else:
                for i in range(testData.size(0)):
                    t = tensor_to_sent(testData[i][k, :].unsqueeze(0).data.cpu().numpy(), inv_dict, True)
                    print(t)
                    fout.write(str(t[0]) + '\n')

                print("gold:", gt[0])
                fout.write("gold: " + str(gt[0]) + '\n')
                print("pred:", pt[0][0])
                fout.write("pred: " + str(pt[0][0]) + '\n\n')

                print()

    model.dec.set_teacher_forcing(cur_tc)
    fout.close()


def rec_inference_beam(dataloader, model, inv_dict, options):
    criteria = nn.CrossEntropyLoss(ignore_index=0, size_average=False)
    if use_cuda:
        criteria.cuda()

    cur_tc = model.dec.get_teacher_forcing()
    model.dec.set_teacher_forcing(True)
    fout = open(options.name + "_result5.txt", 'w', encoding='utf-8')
    model.load_state_dict(torch.load(options.name))
    model.eval()

    for i_batch, sample_batch in enumerate(dataloader):
        if sample_batch.size(0) < 1:
            continue
        testData = sample_batch[:1]
        gold = sample_batch[-1]

        us = testData.transpose(0, 1).contiguous().clone()
        qu_seq = model.base_enc(us.view(-1, us.size(2))).view(us.size(0), us.size(1), -1)
        final_session_o = model.ses_enc(qu_seq, None)

        for k in range(testData.size(1)):
            sent = generate(model, final_session_o[k, :, :].unsqueeze(0), options)
            u = Variable(torch.cuda.LongTensor(sent[0][0])).view(1, 1, -1)
            quseq = model.base_enc(u.view(1, -1)).view(1, 1, -1)
            final_session_o = model.ses_enc(quseq, None)

            sent2 = generate(model, final_session_o[k, :, :].unsqueeze(0), options)

            pt = tensor_to_sent(sent, inv_dict)
            pt2 = tensor_to_sent(sent2, inv_dict)
            gt = tensor_to_sent(gold[k, :].unsqueeze(0).data.cpu().numpy(), inv_dict, True)

            if not options.pretty:
                print(pt)
                print("Ground truth {} {} \n".format(gt, get_sent_ll(gold[k, :].unsqueeze(0), model, criteria,
                                                                     final_session_o)))
            else:
                for i in range(1):
                    t = tensor_to_sent(testData[i][k, :].unsqueeze(0).data.cpu().numpy(), inv_dict, True)
                    print(t)
                    fout.write(str(t[0]) + '\n')

                print("pred:1", pt[0][0])
                fout.write("pred1: " + str(pt[0][0]) + '\n')
                print("pred:2", pt2[0][0])
                fout.write("pred2: " + str(pt2[0][0]) + '\n\n')

                print()

    model.dec.set_teacher_forcing(cur_tc)
    fout.close()


def chat_inference_beam(model, inv_dict, options):
    criteria = nn.CrossEntropyLoss(ignore_index=0, size_average=False)
    if use_cuda:
        criteria.cuda()

    model.dec.set_teacher_forcing(True)
    model.load_state_dict(torch.load(options.name))
    model.eval()

    with open('./data/vocab.pkl', 'rb') as f:
        _ = pickle.load(f)
        w2i = pickle.load(f)
    while True:
        s = input()
        if s == 'q':
            print('exit')
            break
        t = [1]
        for word in s:
            if word == ' ':
                t.append(w2i['^'])
            else:
                t.append(w2i[word])

        t.append(2)
        temp = Variable(torch.cuda.LongTensor(t))
        temp = temp.unsqueeze(0).unsqueeze(0).contiguous().clone()
        qu_seq = model.base_enc(temp.view(-1, temp.size(2))).view(temp.size(0), temp.size(1), -1)
        final_session_o = model.ses_enc(qu_seq, None)

        while True:
            sent = generate(model, final_session_o[0, :, :].unsqueeze(0), options)

            pt = tensor_to_sent(sent, inv_dict)
            print("response: ", pt[0][0])

            temp = Variable(torch.cuda.LongTensor(sent[0][0]))
            temp = temp.unsqueeze(0).unsqueeze(0).contiguous().clone()
            qu_seq = model.base_enc(temp.view(-1, temp.size(2))).view(temp.size(0), temp.size(1), -1)
            final_session_o = model.ses_enc(qu_seq, final_session_o)

            s = input()
            if s == 'q':
                print('reset')
                break
            t = [1]
            for word in s:
                if word == ' ':
                    t.append(w2i['^'])
                else:
                    t.append(w2i[word])
            t.append(2)
            temp = Variable(torch.cuda.LongTensor(t))
            temp = temp.unsqueeze(0).unsqueeze(0).contiguous().clone()
            qu_seq = model.base_enc(temp.view(-1, temp.size(2))).view(temp.size(0), temp.size(1), -1)
            final_session_o = model.ses_enc(qu_seq, final_session_o)


def calc_valid_loss(data_loader, criteria, model):
    model.eval()
    cur_tc = model.dec.get_teacher_forcing()
    model.dec.set_teacher_forcing(True)
    valid_loss, num_words = 0, 0

    for i_batch, sample_batch in enumerate(tqdm(data_loader)):
        trainData = sample_batch[:-1]
        gold = sample_batch[-1]
        preds, lmpreds = model(trainData, gold)

        preds = preds[:, :-1, :].contiguous().view(-1, preds.size(2))
        gold = gold[:, 1:].contiguous().view(-1)
        # do not include the lM loss, exp(loss) is perplexity
        loss = criteria(preds, gold)
        num_words += gold.ne(0).long().sum().data[0]
        valid_loss += loss.data[0]

    model.train()
    model.dec.set_teacher_forcing(cur_tc)

    return valid_loss / num_words


def data_to_seq():
    _dict_file = '/home/harshals/hed-dlg/Data/MovieTriples/Training.dict.pkl'
    with open(_dict_file, 'rb') as fp2:
        dict_data = pickle.load(fp2)
    inv_dict, vocab_dict = {}, {}
    for x in dict_data:
        tok, f, _, _ = x
        inv_dict[f] = tok
        vocab_dict[tok] = f
    _file = '/data2/chatbot_eval_issues/results/AMT_NCM_Test_NCM_Joao/neural_conv_model_eval_source.txt'
    with open(_file, 'r') as fp:
        all_seqs = []
        for lin in fp.readlines():
            seq = list()
            seq.append(1)
            for wrd in lin.split(" "):
                if wrd not in vocab_dict:
                    seq.append(0)
                else:
                    seq_id = vocab_dict[wrd]
                    seq.append(seq_id)
            seq.append(2)
        all_seqs.append(seq)

    with open('CustomTest.pkl', 'wb') as handle:
        pickle.dump(all_seqs, handle, protocol=pickle.HIGHEST_PROTOCOL)


def uniq_answer(fil):
    uniq = Counter()
    with open(fil + '_result.txt', 'r') as fp:
        all_lines = fp.readlines()
        for line in all_lines:
            resp = line.split("    |    ")
            uniq[resp[1].strip()] += 1
    print('uniq', len(uniq), 'from', len(all_lines))
    print('---all---')
    for s in uniq.most_common():
        print(s)


def main():
    print('torch version {}'.format(torch.__version__))
    with open('./data/vocab.pkl', 'rb') as f:
        inv_dict = pickle.load(f)

    parser = argparse.ArgumentParser(description='HRED parameter options')
    parser.add_argument('-n', dest='name', default='./model/movie_triple_multi_tc_0.4_300_400_400_800_017',
                        help='enter suffix for model files')
    parser.add_argument('-e', dest='epoch', type=int, default=200, help='number of epochs')
    parser.add_argument('-pt', dest='patience', type=int, default=2,
                        help='validtion patience for early stopping default none')
    parser.add_argument('-tc', dest='teacher', action='store_true', default=True, help='default teacher forcing')
    parser.add_argument('-bi', dest='bidi', action='store_true', default=True, help='bidirectional enc/decs')
    parser.add_argument('-test', dest='test', action='store_true', default=True, help='only test or inference')
    parser.add_argument('-shrd_dec_emb', dest='shrd_dec_emb', action='store_true', default=False,
                        help='shared embedding in/out for decoder')
    parser.add_argument('-btstrp', dest='btstrp', default=None, help='bootstrap/load parameters give name')
    parser.add_argument('-toy', dest='toy', action='store_true', default=False,
                        help='loads only 1000 training and 100 valid for testing')
    parser.add_argument('-pretty', dest='pretty', action='store_true', default=True, help='pretty print inference')
    parser.add_argument('-mmi', dest='mmi', action='store_true', default=False,
                        help='Using the mmi anti-lm for ranking beam')
    parser.add_argument('-drp', dest='drp', type=float, default=0.4, help='dropout probability used all throughout')
    parser.add_argument('-nl', dest='num_lyr', type=int, default=1, help='number of enc/dec layers(same for both)')
    parser.add_argument('-lr', dest='lr', type=float, default=0.0001, help='learning rate for optimizer')
    parser.add_argument('-bs', dest='bt_siz', type=int, default=32, help='batch size')
    parser.add_argument('-bms', dest='beam', type=int, default=1, help='beam size for decoding')
    parser.add_argument('-vsz', dest='vocab_size', type=int, default=len(inv_dict), help='size of vocabulary')
    parser.add_argument('-esz', dest='emb_size', type=int, default=300, help='embedding size enc/dec same')
    parser.add_argument('-uthid', dest='ut_hid_size', type=int, default=400, help='encoder utterance hidden state')
    parser.add_argument('-seshid', dest='ses_hid_size', type=int, default=400, help='encoder session hidden state')
    parser.add_argument('-dechid', dest='dec_hid_size', type=int, default=800, help='decoder hidden state')

    options = parser.parse_args()
    print(options)

    options.vocab_size = len(inv_dict)
    model = Seq2Seq(options)
    if use_cuda:
        model.cuda()

    if not options.test:
        train(options, model)
    else:
        if options.toy:
            test_dataset = MovieTriples('test', 100)
        else:
            test_dataset = MovieTriples('test', options.bt_siz)

        test_dataloader = DataLoader(test_dataset, 1, shuffle=True, collate_fn=custom_collate_fn)
        inference_beam(test_dataloader, model, inv_dict, options)
        # chat_inference_beam(model, inv_dict, options)


main()
