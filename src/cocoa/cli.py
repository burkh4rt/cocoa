#!/usr/bin/env python3

"""
CLI for cocoa - configurablecollation and tokenization of clinical data
"""

import pathlib
from typing import Annotated, Optional

import typer
from rich import print
from rich.console import Console
from rich.table import Table

from cocoa.collator import Collator
from cocoa.tokenizer import Tokenizer

app = typer.Typer(
    name="cocoa", help="Configurable collation and tokenization", add_completion=False
)
console = Console()


@app.command()
def collate(
    output: Annotated[
        Optional[pathlib.Path],
        typer.Option("--output", "-o", help="Output directory for collated data"),
    ] = None,
):
    """
    Collate raw data into a denormalized format.

    Reads configuration from config/main.yaml and produces a MEDS-like
    parquet file with collated events.
    """
    with console.status("[bold green]Collating data..."):
        collator = Collator()
        collator.save_all(path=output)
    print("[green]✓[/green] Collation complete")
    out_path = output or collator.cfg.processed_data_home
    print(f"  Output: [cyan]{out_path}/meds.parquet[/cyan]")


@app.command()
def tokenize(
    output: Annotated[
        Optional[pathlib.Path],
        typer.Option("--output", "-o", help="Output directory for tokenized data"),
    ] = None,
):
    """
    Tokenize collated data into integer sequences.

    Reads collated parquet files and produces tokenized timelines with
    vocabulary and bin information.
    """
    with console.status("[bold green]Tokenizing data..."):
        tokenizer = Tokenizer()
        tokenizer.save_all(path=output)
    print("[green]✓[/green] Tokenization complete")
    out_path = output or tokenizer.cfg.processed_data_home
    print(f"  Output: [cyan]{out_path}/tokens_times.parquet[/cyan]")
    print(f"  Vocabulary size: [cyan]{len(tokenizer)}[/cyan] tokens")


@app.command()
def pipeline(
    output: Annotated[
        Optional[pathlib.Path],
        typer.Option("--output", "-o", help="Output directory for all outputs"),
    ] = None,
):
    """
    Run the full pipeline: collate and tokenize.
    """
    print("[bold]Running full pipeline[/bold]\n")
    collate(output=output)
    tokenize(output=output)
    print("\n[bold green]Pipeline complete![/bold green]")


@app.command()
def info():
    """
    Display configuration information.
    """
    from omegaconf import OmegaConf

    main_cfg = OmegaConf.load(pathlib.Path("./config/main.yaml").expanduser().resolve())

    table = Table(title="Cocoa Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Data Home", str(main_cfg.data_home))
    table.add_row("Processed Data Home", str(main_cfg.processed_data_home))
    table.add_row("Collation Config", str(main_cfg.collation_config))
    table.add_row("Tokenization Config", str(main_cfg.tokenization_config))

    console.print(table)


@app.command()
def test():
    """
    Run the test suite.
    """
    import subprocess
    import sys

    # Always run pytest with verbose output by default
    args = [sys.executable, "-m", "pytest", "tests/", "-vv", "-r", "a", "-s"]
    raise SystemExit(subprocess.call(args))


def main():
    app()


if __name__ == "__main__":
    main()
