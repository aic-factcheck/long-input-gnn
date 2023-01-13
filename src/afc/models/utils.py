import torch
from torch.utils.data import DataLoader
from transformers.modeling_outputs import SequenceClassifierOutput
from rich.progress import Progress
from typing import Tuple


def evaluate_model(
    model: torch.nn.Module, dataloader: DataLoader, display_progress=True
) -> Tuple[float, float]:
    assert hasattr(
        model, "_common_step"
    ), "The model does not have the common step function"

    model.eval()
    total_loss, total_accuracy = 0, 0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)

    with torch.no_grad(), Progress() as progress:
        if display_progress:
            task = progress.add_task("Evaluating the model...", total=len(dataloader))

        for batch_idx, batch in enumerate(dataloader):
            batch = {
                k: v.to(dev) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch_loss, batch_accuracy, batch_predictions = model._common_step(
                batch, batch_idx, stage="train"
            )
            total_loss += batch_loss.item()
            total_accuracy += batch_accuracy

            if display_progress:
                progress.update(task, advance=1)

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)

    return avg_loss, avg_accuracy


def get_model_predictions(model: torch.nn.Module, dataloader: DataLoader, display_progress=True):
    model.eval()
    predictions = None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)

    with torch.no_grad(), Progress() as progress:
        if display_progress:
            task = progress.add_task("Getting model predictions...", total=len(dataloader))

        for batch_idx, batch in enumerate(dataloader):
            batch = {
                k: v.to(dev) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            batch_predictions = model(**batch)
            if isinstance(batch, SequenceClassifierOutput):
                batch_predictions = batch_predictions.logits
            
            batch_predictions = batch_predictions.argmax(dim=1)

            if predictions is None:
                predictions = batch_predictions
            else:
                predictions = torch.concat((predictions, batch_predictions), dim=0)

            if display_progress:
                progress.update(task, advance=1)

    return predictions
