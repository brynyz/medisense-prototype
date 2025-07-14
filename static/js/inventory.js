$(document).ready(function() {
  $('[data-toggle="tooltip"]').tooltip();

  // Open popup for Add or Edit
  $(".popup-btn").on("click", function(e) {
    e.preventDefault();
    $(".popup").addClass("active");

    // If this is an edit button, set the form action and populate fields
    var editUrl = $(this).data("edit-url");
    var itemId = $(this).data("item-id");
    if (editUrl && itemId) {
      $(".popup-form").attr("action", editUrl);
    } else {
      // Reset form for Add
      $(".popup-form")[0].reset();
    }
  });

  $(".popup .close-btn").on("click", function() {
    $(".popup").removeClass("active");
  });
});