from utils.utils import AverageMeter, accuracy
import numpy as np
import torch
import time
import logging
import csv


def apply_category(category, target):
    classes = [category[t] for t in target.flatten()]
    classes = np.array(classes).reshape(target.shape).tolist()
    return classes


def validate(val_loader, model, criterion, print_freq):
    log = logging.getLogger('my')
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target, path) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                '''
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
                '''

                log.info('Test: [{0}/{1}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        log.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                 .format(top1=top1, top5=top5))

        return top1.avg


def train(train_loader, model, criterion, optimizer, epoch, print_freq):
    log = logging.getLogger('my')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, path) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            '''
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            '''
            log.info('Epoch: [{0}][{1}/{2}]\t'
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                     'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time
                , data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate_video(val_loader, model, criterion, print_freq):
    log = logging.getLogger('my')
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    rank_target = AverageMeter()
    score_target = AverageMeter()
    prob_target = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target, path) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # compute top5 score, prob and target rank, score, prob
            scroe_top5, _ = output.data.topk(k=5)
            score = output.data[:, target].diag().reshape((-1, 1))
            prob = torch.nn.functional.softmax(output.data, dim=1)
            prob_top5, cls_top5 = prob.topk(k=47)
            rank = cls_top5.eq(target.reshape(-1, 1)).argmax(dim=1, keepdim=True)
            prob_top5 = prob_top5[:, :5]
            cls_top5 = cls_top5[:, :5]
            prob = prob[:, target].diag().reshape((-1, 1))

            rank_target.update(torch.mean(rank.type(torch.float32), dim=0).item(), input.size(0))
            prob_target.update(torch.mean(prob, dim=0).item(), input.size(0))
            score_target.update(torch.mean(score, dim=0).item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                '''
                print('Test-Video : [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'Rank {rank.val:.2f} ({rank.avg:.2f})\t'
                      'Prob {prob.val:.5f} ({prob.avg:.5f})\t'
                      'Score {score.val:.5f} ({score.avg:.5f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5, rank=rank_target, prob=prob_target, score=score_target))
                '''
                log.info('Test-Video : [{0}/{1}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                         'Rank {rank.val:.2f} ({rank.avg:.2f})\t'
                         'Prob {prob.val:.5f} ({prob.avg:.5f})\t'
                         'Score {score.val:.5f} ({score.avg:.5f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5, rank=rank_target, prob=prob_target, score=score_target))
                '''
                for n, info in enumerate(zip(target, path, scene, rank, prob, score, cls_top5, prob_top5, scroe_top5)):
                    print(info)
                    if n == 10:
                        break
                '''
        log.info(" * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}\t"
                 " At target\tRank {rank.avg:.2f}\tProb {prob.avg:.5f}\tScore {score.avg:.5f}"
                 .format(top1=top1, top5=top5, rank=rank_target, prob=prob_target, score=score_target))
        '''
        print(" * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}\t"
              " At target\tRank {rank.avg:.2f}\tProb {prob.avg:.5f}\tScore {score.avg:.5f}"
              .format(top1=top1, top5=top5, rank=rank_target, prob=prob_target, score=score_target))
        '''
        return rank_target.avg, prob_target.avg, score_target.avg


def validate_video_csv(val_loader, model, criterion, category, csv_file, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    rank_target = AverageMeter()
    score_target = AverageMeter()
    prob_target = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    f = open(csv_file, 'w', newline='', encoding='utf-8')
    wrt = csv.writer(f)
    wrt.writerow(['path', 'target', 'rank', 'prob', 'score', 'top5', 'top5_prob', 'top5_score'])

    with torch.no_grad():
        for i, (input, target, path) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # compute top5 score, prob and target rank, score, prob
            scroe_top5, _ = output.data.topk(k=5)
            score = output.data[:, target].diag().reshape((-1, 1))
            prob = torch.nn.functional.softmax(output.data, dim=1)
            prob_top5, cls_top5 = prob.topk(k=47)
            rank = cls_top5.eq(target.reshape(-1, 1)).argmax(dim=1, keepdim=True)
            prob_top5 = prob_top5[:, :5]
            cls_top5 = cls_top5[:, :5]

            category_target = apply_category(category, target)
            category_top5 = apply_category(category, cls_top5)
            prob = prob[:, target].diag().reshape((-1, 1))

            rank_target.update(torch.mean(rank.type(torch.float32), dim=0).item(), input.size(0))
            prob_target.update(torch.mean(prob, dim=0).item(), input.size(0))
            score_target.update(torch.mean(score, dim=0).item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            for n, info in enumerate(
                    zip(path, category_target, rank.tolist(), prob.tolist(), score.tolist(),
                        category_top5, prob_top5.tolist(), scroe_top5.tolist())):
                info = [info[0], info[1], info[2][0], round(info[3][0], 4), round(info[4][0], 4),
                        ' '.join(info[5]),
                        ' '.join(list(map(lambda x: str(round(x, 4)), info[6]))),
                        ' '.join(list(map(lambda x: str(round(x, 4)), info[7])))]
                wrt.writerow(info)

            if i % print_freq == 0:
                print('Test-Video : [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'Rank {rank.val:.2f} ({rank.avg:.2f})\t'
                      'Prob {prob.val:.5f} ({prob.avg:.5f})\t'
                      'Score {score.val:.5f} ({score.avg:.5f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5, rank=rank_target, prob=prob_target, score=score_target))

        print(" * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}\t"
              " At target\tRank {rank.avg:.2f}\tProb {prob.avg:.5f}\t"
              "Score {score.avg:.5f}"
              .format(top1=top1, top5=top5, rank=rank_target, prob=prob_target, score=score_target))
        return rank_target.avg, prob_target.avg, score_target.avg
