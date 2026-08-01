"""Thin public client; music generation stays in the shared TheArtist backend."""

from .client import ArtistClient, ArtistClientError

__all__ = ["ArtistClient", "ArtistClientError"]
