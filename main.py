from util import *
from model import *
from config import * 


def main():
    seed_all(SEED)

    data = sio.loadmat('./data/ACM.mat')
    
    hg = dgl.heterograph({
        ('paper', 'written-by', 'author') : data['PvsA'].nonzero(),
        ('author', 'writing', 'paper') : data['PvsA'].transpose().nonzero(),
        ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
        ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
        ('paper', 'is-about', 'subject') : data['PvsL'].nonzero(),
        ('subject', 'has', 'paper') : data['PvsL'].transpose().nonzero(),
    })
    print(hg)

    pvc = data['PvsC'].tocsr()
    p_selected = pvc.tocoo()
    
    labels = pvc.indices
    labels = torch.tensor(labels, dtype=torch.int64)
    labels_np = labels.numpy()
    labels = to_device(labels)

    pid = p_selected.row
    assert len(pid) == 12499
    shuffle = np.random.permutation(pid)
    train_idx = shuffle[0:800]
    val_idx = shuffle[800:900]
    test_idx = shuffle[900:]

    for ntype in hg.ntypes:
        emb = torch.empty(hg.number_of_nodes(ntype), INPUT_DIM)
        nn.init.xavier_uniform_(emb)
        hg.nodes[ntype].data['inp'] = emb

    hg = to_device(hg)

    model = HGT(hg,
                in_dim=INPUT_DIM,
                hidden_dim=HIDDEN_DIM,
                out_dim=int(labels.max())+1,
                num_layers=2,
                num_heads=4,
                use_norm=True)
    model = to_device(model)

    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=NUM_EPOCHS, max_lr=MAX_LR)

    best_val_acc = 0.
    best_test_acc = 0.
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        
        logits = model(hg, out_node_type='paper')

        loss = F.cross_entropy(logits[train_idx], labels[train_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        scheduler.step(epoch)
        
        if epoch % 5 == 0:
            model.eval()

            with torch.no_grad():
                logits = model(hg, out_node_type='paper')

            pred = np.argmax(logits.cpu().numpy(), axis=-1) 

            train_acc = np.mean(pred[train_idx] == labels_np[train_idx])
            val_acc = np.mean(pred[val_idx] == labels_np[val_idx])
            test_acc = np.mean(pred[test_idx] == labels_np[test_idx])

            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

            logging.info(f'Epoch: {epoch}, Loss: {float(loss):.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} (Best {best_val_acc:.4f}), Test Acc: {test_acc:.4f} (Best {best_test_acc:.4f})')
            
            
if __name__ == '__main__':
    main() 
