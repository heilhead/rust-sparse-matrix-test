use anyhow::{anyhow, bail, Result};
use std::cmp::Ordering;

pub trait SparseArrayElement: PartialOrd + PartialOrd<usize> {
    fn get_index(&self) -> usize;
}

#[derive(Debug)]
pub struct SparseArray<T: SparseArrayElement> {
    data: Vec<T>,
}

pub struct Iter<'a, T: SparseArrayElement> {
    cur_pos: usize,
    end_idx: Option<usize>,
    data: &'a Vec<T>,
}

pub struct IterMut<'a, T: SparseArrayElement> {
    cur_pos: usize,
    end_idx: Option<usize>,
    data: &'a mut Vec<T>,
}

impl<T: SparseArrayElement> SparseArray<T> {
    pub fn new() -> Self {
        SparseArray { data: Vec::new() }
    }

    pub fn insert(&mut self, elem: T) -> Result<&mut T> {
        let index = elem.get_index();
        let mut data_idx = self.find(index);

        if self.data.len() > 0 {
            if self.data[data_idx] == index {
                bail!("duplicate index: {}", index);
            }

            if self.data[data_idx] < index {
                data_idx += 1;
            }
        }

        self.data.insert(data_idx, elem);

        Ok(&mut self.data[data_idx])
    }

    pub fn remove(&mut self, index: usize) -> Result<()> {
        let data_idx = self.find(index);

        if self.data.len() > 0 && self.data[data_idx] != index {
            bail!("index not found: {}", index);
        }

        self.data.remove(data_idx);

        Ok(())
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if self.data.len() > 0 {
            let data_idx = self.find(index);

            if self.data[data_idx] == index {
                Some(&self.data[data_idx])
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if self.data.len() > 0 {
            let data_idx = self.find(index);

            if self.data[data_idx] == index {
                Some(&mut self.data[data_idx])
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn iter(&self) -> Iter<T> {
        Iter {
            cur_pos: 0,
            end_idx: None,
            data: &self.data,
        }
    }

    pub fn iter_range(&self, start: usize, end: usize) -> Iter<T> {
        Iter {
            cur_pos: self.find(start),
            end_idx: Some(end),
            data: &self.data,
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut {
            cur_pos: 0,
            end_idx: None,
            data: &mut self.data,
        }
    }

    pub fn iter_range_mut(&mut self, start: usize, end: usize) -> IterMut<T> {
        IterMut {
            cur_pos: self.find(start),
            end_idx: Some(end),
            data: &mut self.data,
        }
    }

    pub fn has(&self, index: usize) -> bool {
        if self.data.len() > 0 {
            let data_idx = self.find(index);

            self.data[data_idx] == index
        } else {
            false
        }
    }

    fn find(&self, index: usize) -> usize {
        if self.data.len() == 0 {
            return 0;
        }

        let mut left = 0;
        let mut right = self.data.len() - 1;

        while left != right {
            let middle = ((left + right) as f32 / 2.0).ceil() as usize;

            if self.data[middle] > index {
                right = middle - 1;
            } else {
                left = middle;
            }
        }

        left
    }
}

#[derive(Debug)]
pub struct SparseMatrixRow<T> {
    index: usize,
    cells: SparseArray<SparseMatrixCell<T>>,
}

impl<T> SparseMatrixRow<T> {
    pub fn new(index: usize) -> Self {
        Self {
            index,
            cells: SparseArray::new(),
        }
    }
}

#[derive(Debug)]
pub struct SparseMatrixCell<T> {
    index: usize,
    data: T,
}

impl<T> SparseMatrixCell<T> {
    pub fn new(index: usize, data: T) -> Self {
        Self { index, data }
    }

    pub fn get(&self) -> &T {
        &self.data
    }

    pub fn get_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

#[derive(Debug)]
pub struct SparseMatrix<T> {
    rows: SparseArray<SparseMatrixRow<T>>,
}

impl<T> SparseMatrix<T> {
    pub fn new() -> Self {
        Self {
            rows: SparseArray::new(),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&T, usize, usize)> + '_ {
        self.rows.iter().flat_map(|row| {
            let y = row.index;

            row.cells
                .iter()
                .map(move |cell| (&cell.data, cell.index, y))
        })
    }

    pub fn iter_rect(
        &self,
        left: usize,
        top: usize,
        right: usize,
        bottom: usize,
    ) -> impl Iterator<Item = (&T, usize, usize)> + '_ {
        self.rows.iter_range(top, bottom).flat_map(move |row| {
            let y = row.index;

            row.cells
                .iter_range(left, right)
                .map(move |cell| (&cell.data, cell.index, y))
        })
    }

    pub fn add(&mut self, x: usize, y: usize, data: T) -> Result<()> {
        let cell = SparseMatrixCell::new(x, data);

        match self.rows.get_mut(y) {
            Some(row) => {
                row.cells.insert(cell)?;
            }
            None => {
                let row = self.rows.insert(SparseMatrixRow::new(y))?;
                row.cells.insert(cell)?;
            }
        };

        Ok(())
    }

    pub fn remove(&mut self, x: usize, y: usize) -> Result<()> {
        let row = self
            .rows
            .get_mut(y)
            .ok_or_else(|| anyhow!("row not found: {}", y))?;

        row.cells.remove(x)?;

        if row.cells.data.len() == 0 {
            self.rows.remove(y);
        }

        Ok(())
    }

    pub fn get(&self, x: usize, y: usize) -> Option<&T> {
        self.rows
            .get(y)
            .map(|row| row.cells.get(x).map(|cell| &cell.data))
            .unwrap_or(None)
    }

    pub fn get_mut(&mut self, x: usize, y: usize) -> Option<&mut T> {
        self.rows
            .get_mut(y)
            .map(|row| row.cells.get_mut(x).map(|cell| &mut cell.data))
            .unwrap_or(None)
    }
}

impl<T> SparseArrayElement for SparseMatrixRow<T> {
    fn get_index(&self) -> usize {
        self.index
    }
}

impl<T> SparseArrayElement for SparseMatrixCell<T> {
    fn get_index(&self) -> usize {
        self.index
    }
}

impl<T> PartialEq<usize> for SparseMatrixRow<T> {
    fn eq(&self, other: &usize) -> bool {
        self.index == *other
    }
}

impl<T> PartialOrd<usize> for SparseMatrixRow<T> {
    fn partial_cmp(&self, other: &usize) -> Option<Ordering> {
        Some(self.index.cmp(other))
    }
}

impl<T> PartialEq for SparseMatrixRow<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T> PartialOrd for SparseMatrixRow<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.index.cmp(&other.index))
    }
}

impl<T> PartialEq<usize> for SparseMatrixCell<T> {
    fn eq(&self, other: &usize) -> bool {
        self.index == *other
    }
}

impl<T> PartialOrd<usize> for SparseMatrixCell<T> {
    fn partial_cmp(&self, other: &usize) -> Option<Ordering> {
        Some(self.index.cmp(other))
    }
}

impl<T> PartialEq for SparseMatrixCell<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T> PartialOrd for SparseMatrixCell<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.index.cmp(&other.index))
    }
}

impl<'a, T: SparseArrayElement> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_pos < self.data.len() {
            let pos = self.cur_pos;

            if let Some(end_idx) = self.end_idx {
                if self.data[pos] >= end_idx {
                    self.cur_pos = self.data.len();
                }
            }

            self.cur_pos += 1;

            Some(&self.data[pos])
        } else {
            None
        }
    }
}

impl<'a, T: SparseArrayElement> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_pos < self.data.len() {
            let pos = self.cur_pos;

            if let Some(end_idx) = self.end_idx {
                if self.data[pos] >= end_idx {
                    self.cur_pos = self.data.len();
                }
            }

            self.cur_pos += 1;

            // Safe because we can't get multiple aliases from the same iterator, and the other
            // cases are covered by the regular aliasing rules.
            unsafe { self.data.as_mut_ptr().add(pos).as_mut() }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use super::{SparseMatrix, SparseArray, SparseMatrixCell, SparseMatrixRow};

    #[test]
    fn sparse_array() {
        let mut grid = SparseArray::new();

        assert_eq!(grid.get(1), None);

        grid.insert(SparseMatrixRow {
            index: 5,
            cells: SparseArray::new(),
        });
        grid.insert(SparseMatrixRow {
            index: 3,
            cells: SparseArray::new(),
        });
        grid.insert(SparseMatrixRow {
            index: 4,
            cells: SparseArray::new(),
        });
        grid.insert(SparseMatrixRow {
            index: 1,
            cells: SparseArray::new(),
        });
        grid.insert(SparseMatrixRow {
            index: 2,
            cells: SparseArray::new(),
        });

        assert_eq!(grid.get(1).unwrap().index, 1);
        assert_eq!(grid.get(2).unwrap().index, 2);
        assert_eq!(grid.get(3).unwrap().index, 3);
        assert_eq!(grid.get(4).unwrap().index, 4);
        assert_eq!(grid.get(5).unwrap().index, 5);
        assert_eq!(grid.get(8), None);

        let row = grid.get_mut(1).unwrap();

        row.cells.insert(SparseMatrixCell { index: 5, data: () });
        row.cells.insert(SparseMatrixCell { index: 3, data: () });
        row.cells.insert(SparseMatrixCell { index: 4, data: () });
        row.cells.insert(SparseMatrixCell { index: 1, data: () });
        row.cells.insert(SparseMatrixCell { index: 2, data: () });

        let mut iter = grid.iter();
        assert_eq!(iter.next().unwrap().index, 1);
        assert_eq!(iter.next().unwrap().index, 2);
        assert_eq!(iter.next().unwrap().index, 3);
        assert_eq!(iter.next().unwrap().index, 4);
        assert_eq!(iter.next().unwrap().index, 5);
        assert_eq!(iter.next(), None);

        let mut iter = grid.iter_range(2, 4);
        assert_eq!(iter.next().unwrap().index, 2);
        assert_eq!(iter.next().unwrap().index, 3);
        assert_eq!(iter.next().unwrap().index, 4);
        assert_eq!(iter.next(), None);

        grid.remove(3);
        grid.remove(4);

        let mut iter = grid.iter();
        assert_eq!(iter.next().unwrap().index, 1);
        assert_eq!(iter.next().unwrap().index, 2);
        assert_eq!(iter.next().unwrap().index, 5);
        assert_eq!(iter.next(), None);

        assert_eq!(grid.has(1), true);
        assert_eq!(grid.has(2), true);
        assert_eq!(grid.has(3), false);
        assert_eq!(grid.has(4), false);
        assert_eq!(grid.has(5), true);

        for row in grid.iter_mut() {
            row.cells.insert(SparseMatrixCell { index: 6, data: () });
        }

        assert!(grid.get(1).unwrap().cells.get(6).is_some());
        assert!(grid.get(2).unwrap().cells.get(6).is_some());
        assert!(grid.get(5).unwrap().cells.get(6).is_some());

        // println!("grid: {:?}", grid);
    }

    #[test]
    fn sparse_matrix() {
        #[derive(Debug)]
        struct GridUnit {
            x: usize,
            y: usize,
        }

        let mut grid = SparseMatrix::new();

        for y in 0..5 {
            for x in 0..5 {
                assert!(grid.add(x, y, GridUnit { x, y }).is_ok());
            }

            assert_eq!(grid.rows.get(y).unwrap().cells.data.len(), 5);
        }

        assert_eq!(grid.rows.data.len(), 5);

        for (unit, x, y) in grid.iter_rect(2, 2, 4, 4) {
            println!("pos: {:?} x: {:?} y: {:?}", unit, x, y);
        }

        // println!("grid: {:?}", grid);
    }
}
