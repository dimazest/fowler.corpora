Scenario: Build a vector space from coocurrence matrix
    Given I have co-occurrence counts
    And I have the dictionary from the counts
    And I select the 100 most used tokens as context
    And I select wordsim353 tokens as targets

    When I build a co-occurrence matrix
    And I evaluate the space on the wordsim353 dataset

    Then I should see the evaluation report

Scenario: Apply tf-id weighting to the co-occurrence matrix
    Given I have co-occurrence counts
    And I have the dictionary from the counts
    And I select the 100 most used tokens as context
    And I select wordsim353 tokens as targets

    When I build a co-occurrence matrix
    And I apply tf-idf weighting
    And I evaluate the space on the wordsim353 dataset

    Then I should see the evaluation report

Scenario: Line-normalize the co-occurrence matrix
    Given I have co-occurrence counts
    And I have the dictionary from the counts
    And I select the 100 most used tokens as context
    And I select wordsim353 tokens as targets

    When I build a co-occurrence matrix
    And I line-normalize the matrix
    And I evaluate the space on the wordsim353 dataset

    Then I should see the evaluation report

Scenario: Reduce the co-occurrence matrix using NMF
    Given I have co-occurrence counts
    And I have the dictionary from the counts
    And I select the 100 most used tokens as context
    And I select wordsim353 tokens as targets

    When I build a co-occurrence matrix
    And I line-normalize the matrix
    And I apply NMF
    And I evaluate the space on the wordsim353 dataset

    Then I should see the evaluation report
