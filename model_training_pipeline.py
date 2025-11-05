import torch
from EvolveGCNH import EvolveGCNH
from training import train
from testing import test

model = EvolveGCNH(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_channels'],
            output_dim=output_dim,
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            topk=config['topk']
        ).to(device)

model.reset_parameters()

optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
loss_fn = F.nll_loss

best_model = None
best_valid_acc = 0

for epoch in range(1, 1 + args["epochs"]):
    loss = train(model, data, train_idx, optimizer, loss_fn)
    result = test(model, data, split_idx, evaluator)
    train_acc, valid_acc, test_acc = result
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        best_model = copy.deepcopy(model)
    print(f'Epoch: {epoch:02d}, '
          f'Loss: {loss:.4f}, '
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * valid_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')