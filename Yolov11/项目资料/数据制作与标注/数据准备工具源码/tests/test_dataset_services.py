import unittest
from pathlib import Path

from dataset_tool.services.image_sampler import sample_images
from dataset_tool.services.labeled_splitter import split_labeled_dataset
from tests.support import temporary_directory


class DatasetServiceTests(unittest.TestCase):
    def test_labeled_pairing_seed_and_source_preservation(self):
        with temporary_directory() as root:
            images, labels = root / "images", root / "labels"
            images.mkdir(); labels.mkdir()
            for index in range(10):
                (images / "{}.jpg".format(index)).write_bytes(("image-{}".format(index)).encode("ascii"))
                (labels / "{}.txt".format(index)).write_text("", encoding="utf-8")
            (images / "no-label.jpg").write_bytes(b"orphan")
            output = root / "out"
            result = split_labeled_dataset(images, labels, output, random_seed=7)
            self.assertEqual(result["total_pairs"], 10)
            self.assertEqual(len(result["missing_labels"]), 1)
            self.assertEqual(sum(result["counts"].values()), 10)
            self.assertTrue((images / "0.jpg").is_file())
            self.assertTrue((labels / "0.txt").is_file())
            output2 = root / "out2"
            repeated = split_labeled_dataset(images, labels, output2, random_seed=7)
            self.assertEqual((output / "split_info.txt").read_text(encoding="utf-8"),
                             (output2 / "split_info.txt").read_text(encoding="utf-8"))

    def test_image_sampler_copies_only_images_and_refuses_nonempty_output(self):
        with temporary_directory() as root:
            source = root / "source"
            source.mkdir()
            for index in range(20):
                (source / "{}.jpg".format(index)).write_bytes(b"image")
            result = sample_images(source, root / "sampled", random_seed=42)
            self.assertEqual(result["selected_count"], 4)
            self.assertFalse(result["labels_copied"])
            self.assertEqual(sum(result["counts"].values()), 4)
            self.assertEqual(len(list(source.glob("*.jpg"))), 20)
            with self.assertRaises(FileExistsError):
                sample_images(source, root / "sampled")

    def test_non_finite_ratios_are_rejected(self):
        with temporary_directory() as root:
            images, labels = root / "images", root / "labels"
            images.mkdir(); labels.mkdir()
            (images / "one.jpg").write_bytes(b"image")
            (labels / "one.txt").write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                split_labeled_dataset(images, labels, root / "split", float("nan"), 0.2, 0.1)
            (root / "source").mkdir()
            (root / "source" / "one.jpg").write_bytes(b"image")
            with self.assertRaises(ValueError):
                sample_images(root / "source", root / "sample", float("nan"))


if __name__ == "__main__":
    unittest.main()
